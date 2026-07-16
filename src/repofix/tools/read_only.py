"""Local, read-only tools for inspecting a prepared repository workspace."""

from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Never

from repofix.agent import ToolExecutionError


MAX_LISTING_ENTRIES = 1_000
MAX_READ_FILE_BYTES = 1_000_000
MAX_READ_LINES = 500
MAX_READ_OUTPUT_CHARS = 100_000
MAX_SEARCH_ENTRIES = 10_000
MAX_SEARCH_FILES = 2_000
MAX_SEARCH_MATCHES = 500
MAX_SEARCH_OUTPUT_CHARS = 100_000
MAX_SEARCH_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class _AllowedPath:
    logical: PurePosixPath
    local: Path
    resolved: Path
    is_directory: bool


def _fail(message: str, cause: BaseException | None = None) -> Never:
    if cause is None:
        raise ToolExecutionError(message)
    raise ToolExecutionError(message) from cause


def _parse_relative_path(value: str, *, description: str) -> PurePosixPath:
    if not isinstance(value, str):
        _fail(f"{description} must be a string")
    if not value or not value.strip():
        _fail(f"{description} must not be empty")
    if "\0" in value:
        _fail(f"{description} must not contain NUL bytes")
    if "\\" in value:
        _fail(f"{description} must use POSIX separators")
    if value.startswith("/"):
        _fail(f"{description} must be repository-relative")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        _fail(f"{description} contains invalid path components")
    path = PurePosixPath(value)
    if path.is_absolute():
        _fail(f"{description} must be repository-relative")
    return path


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _logical_contains(root: _AllowedPath, candidate: PurePosixPath) -> bool:
    if candidate == root.logical:
        return True
    return root.is_directory and candidate.is_relative_to(root.logical)


def _matches_glob(path: str, pattern: str) -> bool:
    path_parts = PurePosixPath(path).parts
    pattern_parts = PurePosixPath(pattern).parts
    if len(pattern_parts) == 1:
        return fnmatchcase(path_parts[-1], pattern_parts[0])

    @lru_cache(maxsize=None)
    def matches(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        component = pattern_parts[pattern_index]
        if component == "**":
            return matches(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and matches(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatchcase(path_parts[path_index], component)
            and matches(path_index + 1, pattern_index + 1)
        )

    return matches(0, 0)


class LocalReadOnlyToolGateway:
    """Safely inspect files within configured repository source roots."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        allowed_source_paths: tuple[str, ...],
    ) -> None:
        try:
            resolved_workspace = workspace_root.resolve(strict=True)
        except FileNotFoundError as error:
            _fail("workspace root does not exist", error)
        except (OSError, RuntimeError) as error:
            _fail("workspace root could not be resolved", error)
        if not resolved_workspace.is_dir():
            _fail("workspace root is not a directory")
        if not allowed_source_paths:
            _fail("allowed source paths must not be empty")

        allowed_paths: list[_AllowedPath] = []
        for value in allowed_source_paths:
            logical = _parse_relative_path(value, description="allowed source path")
            local = resolved_workspace.joinpath(*logical.parts)
            try:
                resolved = local.resolve(strict=True)
            except FileNotFoundError as error:
                _fail("an allowed source path does not exist", error)
            except (OSError, RuntimeError) as error:
                _fail("an allowed source path could not be resolved", error)
            if not _contains(resolved_workspace, resolved):
                _fail("an allowed source path resolves outside the workspace")
            allowed_paths.append(
                _AllowedPath(
                    logical=logical,
                    local=local,
                    resolved=resolved,
                    is_directory=resolved.is_dir(),
                )
            )

        self._workspace_root = resolved_workspace
        self._allowed_paths = tuple(allowed_paths)

    def _resolved_is_allowed(self, resolved: Path) -> bool:
        for allowed in self._allowed_paths:
            if resolved == allowed.resolved:
                return True
            if allowed.is_directory and _contains(allowed.resolved, resolved):
                return True
        return False

    def _authorize(self, value: str) -> tuple[Path, Path, PurePosixPath]:
        logical = _parse_relative_path(value, description="tool path")
        if not any(_logical_contains(allowed, logical) for allowed in self._allowed_paths):
            _fail("tool path is outside the allowed source paths")

        local = self._workspace_root.joinpath(*logical.parts)
        parent = self._workspace_root
        for component in logical.parts[:-1]:
            parent /= component
            if parent.is_symlink():
                _fail("tool paths must not traverse symbolic-link directories")
        try:
            resolved = local.resolve(strict=True)
        except FileNotFoundError as error:
            _fail("requested repository path does not exist", error)
        except (OSError, RuntimeError) as error:
            _fail("requested repository path could not be resolved", error)
        if not _contains(self._workspace_root, resolved):
            _fail("requested repository path resolves outside the workspace")
        if not self._resolved_is_allowed(resolved):
            _fail("requested repository path resolves outside the allowed source paths")
        return local, resolved, logical

    def list_files(self, path: str) -> str:
        """List a file or the direct children of a directory deterministically.

        In-boundary symbolic links are shown without a trailing slash and are
        never traversed as directories. Escaping and broken links are omitted.
        """
        local, resolved, logical = self._authorize(path)
        if resolved.is_file():
            return logical.as_posix()
        if not resolved.is_dir():
            _fail("requested repository path is not a file or directory")
        if local.is_symlink():
            _fail("symbolic-link directories cannot be listed")

        entries: list[str] = []
        entry_count = 0
        try:
            for child in local.iterdir():
                entry_count += 1
                if entry_count > MAX_LISTING_ENTRIES:
                    _fail("repository directory exceeds the listing-entry limit")
                child_logical = logical / child.name
                if "\\" in child.name:
                    continue
                if child.is_symlink():
                    try:
                        child_resolved = child.resolve(strict=True)
                    except (FileNotFoundError, OSError, RuntimeError):
                        continue
                    if not _contains(self._workspace_root, child_resolved):
                        continue
                    if not self._resolved_is_allowed(child_resolved):
                        continue
                    entries.append(child_logical.as_posix())
                    continue
                is_directory = child.is_dir()
                suffix = "/" if is_directory else ""
                entries.append(f"{child_logical.as_posix()}{suffix}")
        except OSError as error:
            _fail("repository directory could not be listed", error)
        return "\n".join(sorted(entries))

    def read_file(self, path: str, start_line: int, end_line: int) -> str:
        """Read a one-based inclusive range from an authorized UTF-8 text file."""
        if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line <= 0:
            _fail("start line must be a strict positive integer")
        if isinstance(end_line, bool) or not isinstance(end_line, int) or end_line <= 0:
            _fail("end line must be a strict positive integer")
        if end_line < start_line:
            _fail("end line must not precede start line")
        if end_line - start_line + 1 > MAX_READ_LINES:
            _fail("requested line range exceeds the read limit")

        _, resolved, _ = self._authorize(path)
        if not resolved.is_file():
            _fail("requested repository path is not a regular file")
        try:
            with resolved.open("rb") as file:
                contents = file.read(MAX_READ_FILE_BYTES + 1)
        except OSError as error:
            _fail("repository file could not be read", error)
        if len(contents) > MAX_READ_FILE_BYTES:
            _fail("repository file exceeds the direct-read byte limit")
        if b"\0" in contents:
            _fail("repository file is binary")
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError as error:
            _fail("repository file is not valid UTF-8", error)

        lines = text.splitlines(keepends=True)
        if start_line > len(lines):
            return ""
        selected = lines[start_line - 1 : end_line]
        result = "".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=start_line)
        )
        if len(result) > MAX_READ_OUTPUT_CHARS:
            _fail("rendered file range exceeds the read-output limit")
        return result

    def search_code(self, query: str, file_glob: str | None = None) -> str:
        """Search authorized UTF-8 text files for a literal substring.

        Symlinked, oversized, binary, and undecodable files are skipped.
        """
        if not isinstance(query, str) or not query.strip():
            _fail("search query must be a nonempty string")
        pattern = self._validate_glob(file_glob)
        files = self._collect_search_files()

        matches: list[str] = []
        output_characters = 0
        inspected_files = 0
        for relative_path, local in files:
            if pattern is not None and not _matches_glob(relative_path, pattern):
                continue
            inspected_files += 1
            if inspected_files > MAX_SEARCH_FILES:
                _fail("search exceeds the file-inspection limit")

            contents = self._read_search_candidate(local)
            if contents is None:
                continue
            for line_number, line in enumerate(contents.splitlines(), start=1):
                if query not in line:
                    continue
                match = f"{relative_path}:{line_number}:{line}"
                if len(matches) >= MAX_SEARCH_MATCHES:
                    _fail("search exceeds the match limit")
                added_characters = len(match) + (1 if matches else 0)
                if output_characters + added_characters > MAX_SEARCH_OUTPUT_CHARS:
                    _fail("search exceeds the output-character limit")
                matches.append(match)
                output_characters += added_characters
        return "\n".join(matches)

    def _validate_glob(self, file_glob: str | None) -> str | None:
        if file_glob is None:
            return None
        if not isinstance(file_glob, str) or not file_glob.strip():
            _fail("search glob must be a nonempty string")
        if "\0" in file_glob:
            _fail("search glob must not contain NUL bytes")
        if "\\" in file_glob:
            _fail("search glob must use POSIX separators")
        if file_glob.startswith("/"):
            _fail("search glob must be repository-relative")
        components = file_glob.split("/")
        if any(component in {"", ".", ".."} for component in components):
            _fail("search glob contains invalid path components")
        if PurePosixPath(file_glob).is_absolute():
            _fail("search glob must be repository-relative")
        return file_glob

    def _collect_search_files(self) -> list[tuple[str, Path]]:
        files: dict[str, Path] = {}
        directories: list[tuple[PurePosixPath, Path]] = []
        for allowed in self._allowed_paths:
            if allowed.local.is_symlink():
                continue
            if allowed.is_directory:
                directories.append((allowed.logical, allowed.local))
            elif allowed.resolved.is_file():
                files[allowed.logical.as_posix()] = allowed.local

        entry_count = 0
        while directories:
            logical, directory = directories.pop()
            try:
                for child in directory.iterdir():
                    entry_count += 1
                    if entry_count > MAX_SEARCH_ENTRIES:
                        _fail("search exceeds the repository-entry limit")
                    if child.is_symlink():
                        continue
                    child_logical = logical / child.name
                    if "\\" in child.name:
                        continue
                    if child.is_dir():
                        directories.append((child_logical, child))
                    elif child.is_file():
                        resolved = child.resolve(strict=True)
                        if self._resolved_is_allowed(resolved):
                            files[child_logical.as_posix()] = child
            except ToolExecutionError:
                raise
            except (FileNotFoundError, OSError, RuntimeError) as error:
                _fail("repository search entry could not be inspected", error)
        return sorted(files.items())

    def _read_search_candidate(self, path: Path) -> str | None:
        if path.is_symlink():
            return None
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            _fail("repository search file could not be resolved", error)
        if not _contains(self._workspace_root, resolved) or not self._resolved_is_allowed(resolved):
            return None
        try:
            with path.open("rb") as file:
                contents = file.read(MAX_SEARCH_FILE_BYTES + 1)
        except OSError as error:
            _fail("repository file could not be searched", error)
        if len(contents) > MAX_SEARCH_FILE_BYTES or b"\0" in contents:
            return None
        try:
            return contents.decode("utf-8")
        except UnicodeDecodeError:
            return None
