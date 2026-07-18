"""Tests for local read-only repository tools."""

from collections.abc import Iterator
import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from repofix.agent import ToolExecutionError
from repofix.tools import LocalReadOnlyToolGateway
from repofix.tools import read_only


class RecordingBytesIO(BytesIO):
    """Binary stream that records the bounded read request and result size."""

    requested_size: int | None = None
    returned_size: int | None = None

    def read(self, size: int = -1) -> bytes:
        self.requested_size = size
        contents = super().read(size)
        self.returned_size = len(contents)
        return contents


def create_repository(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "src/app/pkg").mkdir(parents=True)
    (root / "src/app/empty").mkdir()
    (root / "src/application_secret").mkdir()
    (root / "tests").mkdir()
    (root / "src/app/a.py").write_text(
        "alpha\ndef target():\n    return 'target'\n", encoding="utf-8"
    )
    (root / "src/app/b.txt").write_text("target notes\n", encoding="utf-8")
    (root / "src/app/pkg/c.py").write_text("target = 'nested'\n", encoding="utf-8")
    (root / "src/application_secret/secret.py").write_text("target secret\n", encoding="utf-8")
    (root / "tests/test_a.py").write_text("def test_target():\n    pass\n", encoding="utf-8")
    return root


def gateway(root: Path) -> LocalReadOnlyToolGateway:
    return LocalReadOnlyToolGateway(
        workspace_root=root,
        allowed_source_paths=("src/app", "tests"),
    )


def create_symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symbolic links are not supported on this host: {error}")


def test_valid_initialization_allows_configured_root_and_descendant(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    tools = gateway(root)

    assert "src/app/a.py" in tools.list_files("src/app")
    assert tools.list_files("src/app/a.py") == "src/app/a.py"


def test_rejects_missing_workspace_root(tmp_path: Path) -> None:
    with pytest.raises(ToolExecutionError, match="does not exist") as caught:
        gateway(tmp_path / "missing")

    assert str(tmp_path) not in str(caught.value)


def test_rejects_workspace_root_that_is_not_directory(tmp_path: Path) -> None:
    workspace_file = tmp_path / "workspace"
    workspace_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ToolExecutionError, match="not a directory"):
        LocalReadOnlyToolGateway(
            workspace_root=workspace_file, allowed_source_paths=("src",)
        )


def test_rejects_empty_allowed_source_paths(tmp_path: Path) -> None:
    root = create_repository(tmp_path)

    with pytest.raises(ToolExecutionError, match="must not be empty"):
        LocalReadOnlyToolGateway(workspace_root=root, allowed_source_paths=())


def test_rejects_missing_allowed_source_path(tmp_path: Path) -> None:
    root = create_repository(tmp_path)

    with pytest.raises(ToolExecutionError, match="does not exist"):
        LocalReadOnlyToolGateway(
            workspace_root=root, allowed_source_paths=("src/missing",)
        )


@pytest.mark.parametrize(
    "path", ["", "   ", "/src/app", r"src\app", "src/../app", "src/./app", "src//app", "src/\0app"]
)
def test_rejects_invalid_allowed_source_paths(tmp_path: Path, path: str) -> None:
    root = create_repository(tmp_path)

    with pytest.raises(ToolExecutionError):
        LocalReadOnlyToolGateway(workspace_root=root, allowed_source_paths=(path,))


def test_rejects_allowed_path_symlink_escape(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    create_symlink(root / "escaped", outside, directory=True)

    with pytest.raises(ToolExecutionError, match="outside the workspace"):
        LocalReadOnlyToolGateway(workspace_root=root, allowed_source_paths=("escaped",))


def test_rejects_textual_prefix_boundary_and_other_disallowed_paths(tmp_path: Path) -> None:
    tools = gateway(create_repository(tmp_path))

    with pytest.raises(ToolExecutionError, match="outside the allowed"):
        tools.list_files("src/application_secret")
    with pytest.raises(ToolExecutionError, match="outside the allowed"):
        tools.list_files("src")


@pytest.mark.parametrize(
    "path",
    [
        "/src/app/a.py",
        "src/app/../application_secret/secret.py",
        "src/app/./a.py",
        r"src\app\a.py",
        "src/app/a.py\0hidden",
        "   ",
    ],
)
def test_rejects_invalid_tool_paths(tmp_path: Path, path: str) -> None:
    tools = gateway(create_repository(tmp_path))

    with pytest.raises(ToolExecutionError):
        tools.list_files(path)


def test_rejects_direct_and_nested_symlink_escapes(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("external secret\n", encoding="utf-8")
    create_symlink(root / "src/app/external.py", outside / "secret.py")
    create_symlink(root / "src/app/external_dir", outside, directory=True)
    tools = gateway(root)

    with pytest.raises(ToolExecutionError, match="outside the workspace"):
        tools.read_file("src/app/external.py", 1, 1)
    with pytest.raises(ToolExecutionError, match="symbolic-link directories"):
        tools.read_file("src/app/external_dir/secret.py", 1, 1)


def test_symlinked_directories_are_represented_but_not_traversed(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    create_symlink(root / "src/app/pkg_link", root / "src/app/pkg", directory=True)
    tools = gateway(root)

    listing = tools.list_files("src/app").splitlines()

    assert "src/app/pkg_link" in listing
    assert "src/app/pkg_link/" not in listing
    with pytest.raises(ToolExecutionError, match="cannot be listed"):
        tools.list_files("src/app/pkg_link")
    with pytest.raises(ToolExecutionError, match="must not traverse"):
        tools.read_file("src/app/pkg_link/c.py", 1, 1)
    assert "pkg_link" not in tools.search_code("target")


def test_nested_symlink_directory_cannot_be_traversed(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    (root / "src/app/outer").mkdir()
    create_symlink(root / "src/app/outer/pkg_link", root / "src/app/pkg", directory=True)
    tools = gateway(root)

    with pytest.raises(ToolExecutionError, match="must not traverse"):
        tools.read_file("src/app/outer/pkg_link/c.py", 1, 1)


def test_in_boundary_final_symlink_file_uses_resolved_target(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    create_symlink(root / "src/app/a_link.py", root / "src/app/a.py")
    tools = gateway(root)

    assert tools.read_file("src/app/a_link.py", 1, 1) == "1: alpha\n"


def test_external_symlinks_do_not_appear_in_listing_or_search(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("external_unique_match\n", encoding="utf-8")
    create_symlink(root / "src/app/outside.py", outside)
    tools = gateway(root)

    assert "outside.py" not in tools.list_files("src/app")
    assert tools.search_code("external_unique_match") == ""


def test_directory_listing_is_direct_sorted_and_marks_directories(tmp_path: Path) -> None:
    tools = gateway(create_repository(tmp_path))

    assert tools.list_files("src/app") == "\n".join(
        [
            "src/app/a.py",
            "src/app/b.txt",
            "src/app/empty/",
            "src/app/pkg/",
        ]
    )


def test_listing_empty_directory_returns_empty_string(tmp_path: Path) -> None:
    tools = gateway(create_repository(tmp_path))

    assert tools.list_files("src/app/empty") == ""


def test_listing_missing_path_raises_operational_error(tmp_path: Path) -> None:
    tools = gateway(create_repository(tmp_path))

    with pytest.raises(ToolExecutionError, match="does not exist"):
        tools.list_files("src/app/missing.py")


def test_listing_entry_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tools = gateway(create_repository(tmp_path))
    monkeypatch.setattr(read_only, "MAX_LISTING_ENTRIES", 1)

    with pytest.raises(ToolExecutionError, match="listing-entry limit"):
        tools.list_files("src/app")


def test_reads_one_based_inclusive_lines_and_preserves_numbers(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    path = root / "src/app/lines.txt"
    path.write_text("first\nsecond\nthird", encoding="utf-8")
    tools = gateway(root)

    assert tools.read_file("src/app/lines.txt", 2, 3) == "2: second\n3: third"
    assert tools.read_file("src/app/lines.txt", 3, 3) == "3: third"
    assert tools.read_file("src/app/lines.txt", 4, 5) == ""


def test_read_metadata_hashes_complete_file_not_rendered_excerpt(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    contents = (root / "src/app/a.py").read_bytes()

    result = gateway(root).read_file_with_metadata("src/app/a.py", 2, 2)

    assert result.output == "2: def target():\n"
    assert result.full_file_sha256 == hashlib.sha256(contents).hexdigest()
    assert result.full_file_sha256 != hashlib.sha256(result.output.encode()).hexdigest()


def test_read_file_rejects_directory_binary_and_invalid_utf8(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    (root / "src/app/binary.bin").write_bytes(b"text\0binary")
    (root / "src/app/invalid.txt").write_bytes(b"\xff\xfe")
    tools = gateway(root)

    with pytest.raises(ToolExecutionError, match="regular file"):
        tools.read_file("src/app/pkg", 1, 1)
    with pytest.raises(ToolExecutionError, match="binary"):
        tools.read_file("src/app/binary.bin", 1, 1)
    with pytest.raises(ToolExecutionError, match="UTF-8"):
        tools.read_file("src/app/invalid.txt", 1, 1)


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    [
        (True, 1),
        (0, 1),
        (-1, 1),
        (1.5, 2),
        (1, "2"),
        (2, 1),
        (1, read_only.MAX_READ_LINES + 1),
    ],
)
def test_read_file_rejects_invalid_or_excessive_ranges(
    tmp_path: Path, start_line: object, end_line: object
) -> None:
    tools = gateway(create_repository(tmp_path))

    with pytest.raises(ToolExecutionError):
        tools.read_file(  # type: ignore[arg-type]
            "src/app/a.py", start_line=start_line, end_line=end_line
        )


def test_read_file_does_not_modify_contents(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    path = root / "src/app/a.py"
    before = path.read_bytes()

    gateway(root).read_file("src/app/a.py", 1, 3)

    assert path.read_bytes() == before


def test_read_file_uses_bounded_binary_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = create_repository(tmp_path)
    tools = gateway(root)
    stream = RecordingBytesIO(b"x" * 32)
    expected_path = (root / "src/app/a.py").resolve()
    monkeypatch.setattr(read_only, "MAX_READ_FILE_BYTES", 8)

    def open_stream(path: Path, mode: str) -> RecordingBytesIO:
        assert path == expected_path
        assert mode == "rb"
        return stream

    monkeypatch.setattr(Path, "open", open_stream)

    with pytest.raises(ToolExecutionError, match="byte limit"):
        tools.read_file("src/app/a.py", 1, 1)

    assert stream.requested_size == 9
    assert stream.returned_size == 9


def test_read_file_rejects_above_byte_limit_but_accepts_exact_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = create_repository(tmp_path)
    exact = root / "src/app/exact.txt"
    oversized = root / "src/app/oversized-read.txt"
    exact.write_bytes(b"12345678")
    oversized.write_bytes(b"123456789")
    tools = gateway(root)
    monkeypatch.setattr(read_only, "MAX_READ_FILE_BYTES", 8)

    assert tools.read_file("src/app/exact.txt", 1, 1) == "1: 12345678"
    with pytest.raises(ToolExecutionError, match="byte limit"):
        tools.read_file("src/app/oversized-read.txt", 1, 1)


def test_read_file_rejects_excessive_rendered_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tools = gateway(create_repository(tmp_path))
    monkeypatch.setattr(read_only, "MAX_READ_OUTPUT_CHARS", 5)

    with pytest.raises(ToolExecutionError, match="read-output limit"):
        tools.read_file("src/app/a.py", 1, 1)


def test_literal_search_is_deterministic_with_multiple_matches(tmp_path: Path) -> None:
    tools = gateway(create_repository(tmp_path))

    assert tools.search_code("target").splitlines() == [
        "src/app/a.py:2:def target():",
        "src/app/a.py:3:    return 'target'",
        "src/app/b.txt:1:target notes",
        "src/app/pkg/c.py:1:target = 'nested'",
        "tests/test_a.py:1:def test_target():",
    ]


def test_search_supports_simple_and_recursive_globs(tmp_path: Path) -> None:
    tools = gateway(create_repository(tmp_path))

    assert tools.search_code("target", "tests/test_*.py") == (
        "tests/test_a.py:1:def test_target():"
    )
    recursive_results = tools.search_code("target", "src/**/*.py")
    assert "src/app/a.py" in recursive_results
    assert "src/app/pkg/c.py" in recursive_results
    assert "tests/test_a.py" not in recursive_results


def test_search_returns_empty_string_when_no_matches_exist(tmp_path: Path) -> None:
    assert gateway(create_repository(tmp_path)).search_code("not-present-anywhere") == ""


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(tmp_path: Path, query: str) -> None:
    with pytest.raises(ToolExecutionError, match="query"):
        gateway(create_repository(tmp_path)).search_code(query)


@pytest.mark.parametrize(
    "file_glob",
    ["", "   ", "/src/*.py", r"src\*.py", "src/../*.py", "src/./*.py", "src//*.py", "src/\0*.py"],
)
def test_search_rejects_invalid_glob(tmp_path: Path, file_glob: str) -> None:
    with pytest.raises(ToolExecutionError, match="glob"):
        gateway(create_repository(tmp_path)).search_code("target", file_glob)


def test_search_skips_binary_undecodable_and_oversized_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = create_repository(tmp_path)
    (root / "src/app/binary.bin").write_bytes(b"unique_binary\0match")
    (root / "src/app/invalid.txt").write_bytes(b"unique_invalid\xff")
    (root / "src/app/oversized.txt").write_text("unique_oversized" * 10, encoding="utf-8")
    monkeypatch.setattr(read_only, "MAX_SEARCH_FILE_BYTES", 30)
    tools = gateway(root)

    assert tools.search_code("unique_binary") == ""
    assert tools.search_code("unique_invalid") == ""
    assert tools.search_code("unique_oversized") == ""


def test_search_file_inspection_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tools = gateway(create_repository(tmp_path))
    monkeypatch.setattr(read_only, "MAX_SEARCH_FILES", 1)

    with pytest.raises(ToolExecutionError, match="file-inspection limit"):
        tools.search_code("target")


@pytest.mark.parametrize("file_glob", [None, "*.never"])
def test_search_repository_entry_limit_applies_before_glob_filtering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, file_glob: str | None
) -> None:
    tools = gateway(create_repository(tmp_path))
    monkeypatch.setattr(read_only, "MAX_SEARCH_ENTRIES", 2)

    with pytest.raises(ToolExecutionError, match="repository-entry limit"):
        tools.search_code("target", file_glob)


def test_search_stops_collecting_when_repository_entry_limit_is_exceeded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tools = gateway(create_repository(tmp_path))
    original_iterdir = Path.iterdir
    yielded_entries = 0

    def controlled_iterdir(path: Path) -> Iterator[Path]:
        nonlocal yielded_entries
        for child in original_iterdir(path):
            yielded_entries += 1
            if yielded_entries > 3:
                raise AssertionError("search continued after its entry limit")
            yield child

    monkeypatch.setattr(read_only, "MAX_SEARCH_ENTRIES", 2)
    monkeypatch.setattr(Path, "iterdir", controlled_iterdir)

    with pytest.raises(ToolExecutionError, match="repository-entry limit"):
        tools.search_code("target")

    assert yielded_entries == 3


def test_search_match_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tools = gateway(create_repository(tmp_path))
    monkeypatch.setattr(read_only, "MAX_SEARCH_MATCHES", 1)

    with pytest.raises(ToolExecutionError, match="match limit"):
        tools.search_code("target")


def test_search_output_character_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tools = gateway(create_repository(tmp_path))
    monkeypatch.setattr(read_only, "MAX_SEARCH_OUTPUT_CHARS", 5)

    with pytest.raises(ToolExecutionError, match="output-character limit"):
        tools.search_code("target")


def test_search_does_not_modify_repository_files(tmp_path: Path) -> None:
    root = create_repository(tmp_path)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    gateway(root).search_code("target")

    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before
