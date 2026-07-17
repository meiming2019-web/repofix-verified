"""Tests for safe task specification YAML loading."""

from io import BytesIO
from pathlib import Path

import pytest

from repofix.tasks import (
    AgentTaskSpec,
    TaskSpecLoadError,
    load_agent_task_spec,
    load_evaluator_task_bundle,
)
from repofix.tasks.loader import MAX_TASK_SPEC_BYTES
from repofix.tasks import loader


VALID_BUNDLE_YAML = """\
task:
  task_id: task-001
  repository_url: https://github.com/example/project.git
  pre_fix_commit: 0123456789abcdef0123456789abcdef01234567
  issue_title: Fix the failing behavior
  issue_body: The command produces an incorrect result.
  approved_commands:
    unit_tests:
      argv:
        - pytest
        - -q
  allowed_source_paths:
    - src/repofix
    - tests/unit
  timeout_seconds: 300
hidden_tests:
  commands:
    hidden_tests:
      argv:
        - pytest
        - tests/hidden
gold_patch:
  patch: |-
    diff --git a/file.py b/file.py
    --- a/file.py
    +++ b/file.py
"""

VALID_AGENT_YAML = """\
task_id: task-001
repository_url: https://github.com/example/project.git
pre_fix_commit: 0123456789abcdef0123456789abcdef01234567
issue_title: Fix the failing behavior
issue_body: The command produces an incorrect result.
approved_commands:
  unit_tests:
    argv:
      - pytest
      - -q
allowed_source_paths:
  - src/repofix
  - tests/unit
timeout_seconds: 300
"""


class RecordingBytesIO(BytesIO):
    """In-memory binary file that records the requested read bound."""

    requested_size: int | None = None
    returned_size: int | None = None

    def read(self, size: int = -1) -> bytes:
        self.requested_size = size
        contents = super().read(size)
        self.returned_size = len(contents)
        return contents


def write_yaml(tmp_path: Path, contents: str = VALID_BUNDLE_YAML) -> Path:
    """Write task YAML to a temporary file and return its path."""
    path = tmp_path / "task.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_loads_valid_evaluator_task_bundle(tmp_path: Path) -> None:
    bundle = load_evaluator_task_bundle(write_yaml(tmp_path))

    assert bundle.task.task_id == "task-001"
    assert bundle.hidden_tests.commands["hidden_tests"].argv == ("pytest", "tests/hidden")
    assert bundle.gold_patch.patch.startswith("diff --git")


def test_yaml_lists_load_into_tuple_backed_fields(tmp_path: Path) -> None:
    bundle = load_evaluator_task_bundle(write_yaml(tmp_path))

    assert bundle.task.approved_commands["unit_tests"].argv == ("pytest", "-q")
    assert isinstance(bundle.task.approved_commands["unit_tests"].argv, tuple)
    assert bundle.task.allowed_source_paths == ("src/repofix", "tests/unit")
    assert isinstance(bundle.task.allowed_source_paths, tuple)


def test_load_agent_task_spec_returns_only_agent_model(tmp_path: Path) -> None:
    task = load_agent_task_spec(write_yaml(tmp_path))

    assert type(task) is AgentTaskSpec
    assert task.task_id == "task-001"


def test_loads_valid_agent_only_task_yaml(tmp_path: Path) -> None:
    task = load_agent_task_spec(write_yaml(tmp_path, VALID_AGENT_YAML))

    assert type(task) is AgentTaskSpec
    assert task.task_id == "task-001"
    assert task.allowed_source_paths == ("src/repofix", "tests/unit")


def test_complete_bundle_agent_loader_preserves_evaluator_boundary(tmp_path: Path) -> None:
    task = load_agent_task_spec(write_yaml(tmp_path, VALID_BUNDLE_YAML))
    serialized = task.model_dump()
    rendered = repr(serialized)

    assert type(task) is AgentTaskSpec
    assert set(serialized) == set(AgentTaskSpec.model_fields)
    assert "hidden_tests" not in rendered
    assert "gold_patch" not in rendered
    assert "tests/hidden" not in rendered
    assert "diff --git" not in rendered


@pytest.mark.parametrize(
    "evaluator_data",
    [
        "hidden_tests:\n  commands:\n    hidden_tests:\n      argv: [pytest]\n",
        "gold_patch:\n  patch: secret patch\n",
    ],
    ids=["hidden-tests", "gold-patch"],
)
def test_agent_only_document_rejects_evaluator_fields(
    tmp_path: Path, evaluator_data: str
) -> None:
    contents = f"{VALID_AGENT_YAML}{evaluator_data}"

    with pytest.raises(TaskSpecLoadError, match="model validation"):
        load_agent_task_spec(write_yaml(tmp_path, contents))


def test_top_level_task_key_requires_a_complete_evaluator_bundle(tmp_path: Path) -> None:
    contents = "task:\n  task_id: task-001\n"

    with pytest.raises(TaskSpecLoadError, match="model validation"):
        load_agent_task_spec(write_yaml(tmp_path, contents))


def test_agent_only_document_still_rejects_duplicate_keys(tmp_path: Path) -> None:
    contents = VALID_AGENT_YAML.replace(
        "task_id: task-001\n", "task_id: task-001\ntask_id: task-002\n"
    )

    with pytest.raises(TaskSpecLoadError, match="invalid YAML"):
        load_agent_task_spec(write_yaml(tmp_path, contents))


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (VALID_AGENT_YAML.replace("task-001", "&task task-001"), "anchors"),
        (VALID_AGENT_YAML.replace("task-001", "*missing"), "aliases"),
    ],
)
def test_agent_only_document_still_rejects_yaml_references(
    tmp_path: Path, contents: str, message: str
) -> None:
    with pytest.raises(TaskSpecLoadError, match=message):
        load_agent_task_spec(write_yaml(tmp_path, contents))


def test_agent_serialization_excludes_evaluator_data(tmp_path: Path) -> None:
    serialized = load_agent_task_spec(write_yaml(tmp_path)).model_dump()
    rendered = repr(serialized)

    assert set(serialized) == set(AgentTaskSpec.model_fields)
    assert "hidden_tests" not in serialized
    assert "gold_patch" not in serialized
    assert "tests/hidden" not in rendered
    assert "diff --git" not in rendered


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TaskSpecLoadError, match="not found") as caught:
        load_evaluator_task_bundle(tmp_path / "missing.yaml")

    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_rejects_directory_path(tmp_path: Path) -> None:
    with pytest.raises(TaskSpecLoadError, match="directory") as caught:
        load_evaluator_task_bundle(tmp_path)

    assert isinstance(caught.value.__cause__, IsADirectoryError)


def test_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(TaskSpecLoadError, match="UTF-8") as caught:
        load_evaluator_task_bundle(path)

    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_rejects_file_exceeding_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_bytes(b" " * (MAX_TASK_SPEC_BYTES + 1))

    with pytest.raises(TaskSpecLoadError, match="1 MiB") as caught:
        load_evaluator_task_bundle(path)

    assert caught.value.__cause__ is not None


def test_file_read_is_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stream = RecordingBytesIO(b"x" * 32)
    monkeypatch.setattr(loader, "MAX_TASK_SPEC_BYTES", 8)

    def open_stream(path: Path, mode: str) -> RecordingBytesIO:
        assert path == tmp_path / "virtual.yaml"
        assert mode == "rb"
        return stream

    monkeypatch.setattr(Path, "open", open_stream)

    with pytest.raises(TaskSpecLoadError, match="size limit"):
        load_evaluator_task_bundle(tmp_path / "virtual.yaml")

    assert stream.requested_size == 9
    assert stream.returned_size == 9


def test_file_at_exact_size_limit_is_not_rejected_as_oversized(tmp_path: Path) -> None:
    encoded_yaml = VALID_BUNDLE_YAML.encode("utf-8")
    path = tmp_path / "task.yaml"
    path.write_bytes(encoded_yaml + b" " * (MAX_TASK_SPEC_BYTES - len(encoded_yaml)))

    bundle = load_evaluator_task_bundle(path)

    assert bundle.task.task_id == "task-001"


def test_rejects_malformed_yaml(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "task: [unterminated\n")

    with pytest.raises(TaskSpecLoadError, match="malformed YAML") as caught:
        load_evaluator_task_bundle(path)

    assert caught.value.__cause__ is not None


def test_normalizes_recursion_error_during_yaml_scanning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recursion_error = RecursionError("scanner recursion")

    def recursive_scan(*args: object, **kwargs: object) -> object:
        raise recursion_error

    monkeypatch.setattr(loader.yaml, "scan", recursive_scan)

    with pytest.raises(TaskSpecLoadError, match="nesting is too deep") as caught:
        load_evaluator_task_bundle(write_yaml(tmp_path))

    assert caught.value.__cause__ is recursion_error


def test_normalizes_recursion_error_during_yaml_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recursion_error = RecursionError("constructor recursion")

    def recursive_load(*args: object, **kwargs: object) -> object:
        raise recursion_error

    monkeypatch.setattr(loader.yaml, "load_all", recursive_load)

    with pytest.raises(TaskSpecLoadError, match="nesting is too deep") as caught:
        load_evaluator_task_bundle(write_yaml(tmp_path))

    assert caught.value.__cause__ is recursion_error


@pytest.mark.parametrize("contents", ["", "---\n"])
def test_rejects_empty_yaml_document(tmp_path: Path, contents: str) -> None:
    with pytest.raises(TaskSpecLoadError, match="empty"):
        load_evaluator_task_bundle(write_yaml(tmp_path, contents))


def test_rejects_top_level_sequence(tmp_path: Path) -> None:
    with pytest.raises(TaskSpecLoadError, match="top level must be a mapping"):
        load_evaluator_task_bundle(write_yaml(tmp_path, "- task\n- hidden_tests\n"))


def test_rejects_multiple_yaml_documents(tmp_path: Path) -> None:
    contents = f"{VALID_BUNDLE_YAML}\n---\nsecond: document\n"

    with pytest.raises(TaskSpecLoadError, match="exactly one YAML document"):
        load_evaluator_task_bundle(write_yaml(tmp_path, contents))


def test_rejects_duplicate_top_level_keys(tmp_path: Path) -> None:
    contents = f"{VALID_BUNDLE_YAML}\ntask: {{}}\n"

    with pytest.raises(TaskSpecLoadError, match="invalid YAML") as caught:
        load_evaluator_task_bundle(write_yaml(tmp_path, contents))

    assert caught.value.__cause__ is not None


def test_rejects_duplicate_nested_keys(tmp_path: Path) -> None:
    contents = VALID_BUNDLE_YAML.replace(
        "  task_id: task-001\n", "  task_id: task-001\n  task_id: task-002\n"
    )

    with pytest.raises(TaskSpecLoadError, match="invalid YAML"):
        load_evaluator_task_bundle(write_yaml(tmp_path, contents))


def test_rejects_yaml_alias_usage(tmp_path: Path) -> None:
    contents = "task: *missing\nhidden_tests: {}\ngold_patch: {}\n"

    with pytest.raises(TaskSpecLoadError, match="aliases"):
        load_evaluator_task_bundle(write_yaml(tmp_path, contents))


def test_rejects_unreferenced_yaml_anchor(tmp_path: Path) -> None:
    contents = "task: &task {}\nhidden_tests: {}\ngold_patch: {}\n"

    with pytest.raises(TaskSpecLoadError, match="anchors"):
        load_evaluator_task_bundle(write_yaml(tmp_path, contents))


def test_rejects_unsafe_python_object_tag(tmp_path: Path) -> None:
    contents = "!!python/object/apply:builtins.str ['unsafe']\n"

    with pytest.raises(TaskSpecLoadError, match="invalid YAML"):
        load_evaluator_task_bundle(write_yaml(tmp_path, contents))


def test_rejects_unknown_model_fields(tmp_path: Path) -> None:
    contents = VALID_BUNDLE_YAML.replace("  timeout_seconds: 300\n", "  timeout_seconds: 300\n  extra: true\n")

    with pytest.raises(TaskSpecLoadError, match="model validation") as caught:
        load_evaluator_task_bundle(write_yaml(tmp_path, contents))

    assert caught.value.__cause__ is not None


def test_rejects_model_validation_failure(tmp_path: Path) -> None:
    contents = VALID_BUNDLE_YAML.replace(
        "0123456789abcdef0123456789abcdef01234567", "abc123"
    )

    with pytest.raises(TaskSpecLoadError, match="model validation"):
        load_evaluator_task_bundle(write_yaml(tmp_path, contents))


def test_error_message_does_not_expose_evaluator_secrets(tmp_path: Path) -> None:
    hidden_secret = "do-not-expose-hidden-command"
    patch_secret = "do-not-expose-gold-patch"
    contents = VALID_BUNDLE_YAML.replace("tests/hidden", hidden_secret).replace(
        "diff --git a/file.py b/file.py", patch_secret
    )
    contents = contents.replace("0123456789abcdef0123456789abcdef01234567", "invalid")

    with pytest.raises(TaskSpecLoadError) as caught:
        load_evaluator_task_bundle(write_yaml(tmp_path, contents))

    message = str(caught.value)
    assert hidden_secret not in message
    assert patch_secret not in message
