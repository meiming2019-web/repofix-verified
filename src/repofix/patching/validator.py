"""Bind structured patch drafts to a stable, bounded workspace snapshot."""

import difflib
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Never

from repofix.agent import AgentPhase
from repofix.agent.reproduction_loop import ReproductionAgentRunResult, compute_task_fingerprint
from repofix.patching.models import (
    MAX_PATCH_DIFF_CHARS,
    MAX_REPLACEMENT_CHARS,
    MAX_TOTAL_REPLACEMENT_CHARS,
    PATCH_VALIDATION_SUMMARY,
    PatchEditDraft,
    PatchProposalDraft,
    PatchProposalValidationStatus,
    ValidatedPatchEdit,
    ValidatedPatchFileSnapshot,
    ValidatedPatchProposal,
    compute_proposal_digest,
)
from repofix.reproduction import ReproductionStatus
from repofix.tasks import AgentTaskSpec


MAX_PATCH_SOURCE_FILE_BYTES = 1_000_000
MAX_PATCH_ORIGINAL_LINES = 500


class PatchProposalValidationError(RuntimeError):
    """Raised when a patch proposal cannot be safely bound to the workspace."""


@dataclass(frozen=True)
class _FileSnapshot:
    logical: PurePosixPath
    resolved: Path
    mode: int
    device: int
    inode: int
    link_count: int
    size: int
    mtime_ns: int
    ctime_ns: int
    contents: bytes
    sha256: str


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _fail(message: str, cause: BaseException | None = None) -> Never:
    if cause is None:
        raise PatchProposalValidationError(message)
    raise PatchProposalValidationError(message) from cause


def _require_snapshot_host_support() -> None:
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        _fail("patch snapshot validation requires POSIX O_NOFOLLOW support")


def _require_exact_logical_spelling(workspace: Path, logical: PurePosixPath) -> None:
    directory = workspace
    for component in logical.parts:
        try:
            with os.scandir(directory) as entries:
                exact = any(entry.name == component for entry in entries)
        except OSError as error:
            _fail("patch edit path could not be inspected", error)
        if not exact:
            _fail("patch edit path does not use exact repository spelling")
        directory /= component


def _require_nonsymlink_parents(workspace: Path, logical: PurePosixPath) -> None:
    parent = workspace
    for component in logical.parts[:-1]:
        parent /= component
        try:
            parent_lstat = parent.lstat()
        except OSError as error:
            _fail("patch edit path parent could not be inspected", error)
        if stat.S_ISLNK(parent_lstat.st_mode):
            _fail("patch edit paths must not traverse symbolic-link directories")
        if not stat.S_ISDIR(parent_lstat.st_mode):
            _fail("patch edit path parent is not a directory")


def _metadata(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_safe_file_stat(value: os.stat_result) -> None:
    if not stat.S_ISREG(value.st_mode):
        _fail("patch edit target is not a regular non-symbolic-link file")
    if value.st_nlink != 1:
        _fail("patch edit target must have exactly one hard link")
    if value.st_size > MAX_PATCH_SOURCE_FILE_BYTES:
        _fail("patch edit target exceeds the source-file byte limit")


def _read_snapshot(*, workspace: Path, logical: PurePosixPath) -> _FileSnapshot:
    """Read one logical file through a bounded no-follow descriptor."""
    _require_snapshot_host_support()
    local = workspace.joinpath(*logical.parts)
    descriptor: int | None = None
    try:
        _require_exact_logical_spelling(workspace, logical)
        _require_nonsymlink_parents(workspace, logical)
        pre_lstat = local.lstat()
        pre_resolved = local.resolve(strict=True)
        if not _contains(workspace, pre_resolved):
            _fail("patch edit target resolves outside the workspace")

        descriptor = os.open(local, os.O_RDONLY | os.O_NOFOLLOW)
        pre_fstat = os.fstat(descriptor)
        _require_safe_file_stat(pre_fstat)
        if _metadata(pre_lstat) != _metadata(pre_fstat):
            _fail("workspace path identity changed during patch validation")

        chunks: list[bytes] = []
        remaining = MAX_PATCH_SOURCE_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        contents = b"".join(chunks)
        if len(contents) > MAX_PATCH_SOURCE_FILE_BYTES:
            _fail("patch edit target exceeds the source-file byte limit")

        post_fstat = os.fstat(descriptor)
        _require_safe_file_stat(post_fstat)
        _require_exact_logical_spelling(workspace, logical)
        _require_nonsymlink_parents(workspace, logical)
        post_lstat = local.lstat()
        _require_safe_file_stat(post_lstat)
        post_resolved = local.resolve(strict=True)
        if not _contains(workspace, post_resolved):
            _fail("patch edit target resolves outside the workspace")

        expected_metadata = _metadata(pre_lstat)
        if (
            pre_resolved != post_resolved
            or expected_metadata != _metadata(pre_fstat)
            or expected_metadata != _metadata(post_fstat)
            or expected_metadata != _metadata(post_lstat)
            or len(contents) != post_fstat.st_size
        ):
            _fail("workspace path identity changed during patch validation")
    except PatchProposalValidationError:
        raise
    except (FileNotFoundError, OSError, RuntimeError) as error:
        _fail("workspace path identity changed during patch validation", error)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as error:
                _fail("patch snapshot descriptor could not be closed", error)

    return _FileSnapshot(
        logical=logical,
        resolved=pre_resolved,
        mode=pre_lstat.st_mode,
        device=pre_lstat.st_dev,
        inode=pre_lstat.st_ino,
        link_count=pre_lstat.st_nlink,
        size=pre_lstat.st_size,
        mtime_ns=pre_lstat.st_mtime_ns,
        ctime_ns=pre_lstat.st_ctime_ns,
        contents=contents,
        sha256=hashlib.sha256(contents).hexdigest(),
    )


def _verify_snapshot(*, workspace: Path, snapshot: _FileSnapshot) -> None:
    try:
        current = _read_snapshot(workspace=workspace, logical=snapshot.logical)
    except PatchProposalValidationError as error:
        _fail("workspace path identity changed during patch validation", error)
    if current != snapshot:
        _fail("workspace path identity changed during patch validation")


def _line_ending_for(contents: bytes) -> str:
    without_crlf = contents.replace(b"\r\n", b"")
    if b"\r" in without_crlf:
        _fail("patch edit target uses unsupported carriage-return line endings")
    has_crlf = b"\r\n" in contents
    has_lf = b"\n" in without_crlf
    if has_crlf and has_lf:
        _fail("patch edit target uses mixed line endings")
    return "\r\n" if has_crlf else "\n"


def _normalize_replacement(value: str, line_ending: str) -> str:
    if "\r" in value.replace("\r\n", ""):
        _fail("patch replacement uses unsupported carriage-return line endings")
    normalized = value.replace("\r\n", "\n")
    return normalized.replace("\n", line_ending)


def _render_unified_diff(*, path: str, original: str, candidate: str) -> str:
    """Render an applicable standard diff while retaining source terminators."""
    items = difflib.unified_diff(
        original.splitlines(keepends=True),
        candidate.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="\n",
    )
    rendered: list[str] = []
    for item in items:
        if item.startswith((" ", "+", "-")) and not item.endswith("\n"):
            rendered.append(f"{item}\n\\ No newline at end of file\n")
        else:
            rendered.append(item)
    return "".join(rendered)


def _successful_read_hashes(
    reproduction_result: ReproductionAgentRunResult,
) -> dict[str, set[str]]:
    hashes: dict[str, set[str]] = {}
    for observation in reproduction_result.state.observations:
        if not observation.success or observation.tool_name != "read_file":
            continue
        path = observation.arguments.get("path")
        if not isinstance(path, str) or observation.full_file_sha256 is None:
            _fail("successful reproduction reads lack trusted full-file metadata")
        hashes.setdefault(path, set()).add(observation.full_file_sha256)
    return hashes


def validate_patch_workspace_reads(
    *,
    workspace_root: Path,
    task: AgentTaskSpec,
    reproduction_result: ReproductionAgentRunResult,
) -> None:
    """Reject stale patchable files before a patch model is invoked."""
    try:
        workspace = workspace_root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        _fail("workspace root could not be resolved", error)
    if not workspace.is_dir():
        _fail("workspace root is not a directory")
    roots = tuple(PurePosixPath(value) for value in task.patchable_source_paths)
    for path, recorded_hashes in sorted(_successful_read_hashes(reproduction_result).items()):
        logical = PurePosixPath(path)
        if not any(logical == root or logical.is_relative_to(root) for root in roots):
            continue
        if len(recorded_hashes) != 1:
            _fail("reproduction reads contain conflicting full-file hashes")
        snapshot = _read_snapshot(workspace=workspace, logical=logical)
        if snapshot.sha256 != next(iter(recorded_hashes)):
            _fail("patchable source changed after the reproduction read")


def _require_unique_supported_hypothesis(
    reproduction_result: ReproductionAgentRunResult, hypothesis_id: str
) -> None:
    supported = [
        hypothesis.hypothesis_id
        for hypothesis in reproduction_result.state.hypotheses
        if hypothesis.status == "supported"
    ]
    if len(supported) != len(set(supported)):
        _fail("reproduced state contains duplicate supported hypothesis IDs")
    if supported.count(hypothesis_id) != 1:
        _fail("patch proposal hypothesis must identify exactly one supported hypothesis")


def validate_patch_proposal(
    *,
    workspace_root: Path,
    task: AgentTaskSpec,
    reproduction_result: ReproductionAgentRunResult,
    draft: PatchProposalDraft,
) -> ValidatedPatchProposal:
    state = reproduction_result.state
    if state.task_id != task.task_id or state.phase is not AgentPhase.FINISHED:
        _fail("patch proposal does not match a finished reproduction task")
    if reproduction_result.task_fingerprint != compute_task_fingerprint(task):
        _fail("patch proposal task fingerprint does not match reproduction result")
    if (
        not reproduction_result.attempts
        or reproduction_result.attempts[0].verdict.status is not ReproductionStatus.REPRODUCED
    ):
        _fail("patch proposal requires verified reproduction")
    _require_unique_supported_hypothesis(reproduction_result, draft.hypothesis_id)
    if not task.patchable_source_paths:
        _fail("task does not configure patchable source paths")

    try:
        workspace = workspace_root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        _fail("workspace root could not be resolved", error)
    if not workspace.is_dir():
        _fail("workspace root is not a directory")

    grouped: dict[str, list[PatchEditDraft]] = {}
    for edit in draft.edits:
        grouped.setdefault(edit.path, []).append(edit)
    recorded_read_hashes = _successful_read_hashes(reproduction_result)
    patch_roots = tuple(PurePosixPath(value) for value in task.patchable_source_paths)

    preflight_identities: dict[tuple[int, int], PurePosixPath] = {}
    for path_value in sorted(grouped):
        logical = PurePosixPath(path_value)
        if not any(logical == root or logical.is_relative_to(root) for root in patch_roots):
            _fail("patch edit path is outside patchable source paths")
        hashes = recorded_read_hashes.get(path_value)
        if hashes is None:
            _fail("patch edit target was not previously read successfully")
        if len(hashes) != 1:
            _fail("reproduction reads contain conflicting full-file hashes")
        _require_exact_logical_spelling(workspace, logical)
        try:
            preflight_lstat = workspace.joinpath(*logical.parts).lstat()
        except OSError as error:
            _fail("patch edit target could not be inspected", error)
        identity = (preflight_lstat.st_dev, preflight_lstat.st_ino)
        previous = preflight_identities.setdefault(identity, logical)
        if previous != logical:
            _fail("multiple patch paths refer to the same physical file")

    validated: list[ValidatedPatchEdit] = []
    file_snapshots: list[ValidatedPatchFileSnapshot] = []
    diffs: list[str] = []
    snapshots: list[_FileSnapshot] = []
    total_original_lines = 0
    total_effective_replacement_chars = 0

    for path_value in sorted(grouped):
        logical = PurePosixPath(path_value)
        snapshot = _read_snapshot(workspace=workspace, logical=logical)
        recorded_hash = next(iter(recorded_read_hashes[path_value]))
        if snapshot.sha256 != recorded_hash:
            _fail("patch edit target changed after the reproduction read")
        snapshots.append(snapshot)
        file_snapshots.append(
            ValidatedPatchFileSnapshot(
                path=path_value,
                original_file_sha256=snapshot.sha256,
                size_bytes=snapshot.size,
            )
        )

        contents = snapshot.contents
        if b"\0" in contents:
            _fail("patch edit target is binary")
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError as error:
            _fail("patch edit target is not valid UTF-8", error)
        line_ending = _line_ending_for(contents)
        lines = text.splitlines(keepends=True)
        edits = sorted(grouped[path_value], key=lambda item: (item.start_line, item.end_line))
        prior_end = 0
        effective: list[tuple[PatchEditDraft, str]] = []
        for edit in edits:
            if edit.start_line <= prior_end:
                _fail("patch edits overlap within one file")
            if edit.end_line > len(lines):
                _fail("patch edit line range is outside the current file")
            prior_end = edit.end_line
            replacement = _normalize_replacement(edit.replacement_text, line_ending)
            if replacement and edit.end_line < len(lines) and not replacement.endswith(line_ending):
                replacement += line_ending
            if len(replacement) > MAX_REPLACEMENT_CHARS:
                _fail("effective patch replacement exceeds the character limit")
            total_effective_replacement_chars += len(replacement)
            if total_effective_replacement_chars > MAX_TOTAL_REPLACEMENT_CHARS:
                _fail("effective patch replacements exceed the total character limit")
            original = "".join(lines[edit.start_line - 1 : edit.end_line])
            if original == replacement:
                _fail("patch edit replacement is unchanged")
            total_original_lines += edit.end_line - edit.start_line + 1
            validated.append(
                ValidatedPatchEdit(
                    path=path_value,
                    start_line=edit.start_line,
                    end_line=edit.end_line,
                    original_text=original,
                    replacement_text=replacement,
                    original_file_sha256=snapshot.sha256,
                )
            )
            effective.append((edit, replacement))

        candidate = lines[:]
        for edit, replacement in reversed(effective):
            candidate[edit.start_line - 1 : edit.end_line] = [replacement]
        candidate_text = "".join(candidate)
        if candidate_text == "":
            _fail("patch proposals must not empty an entire file")
        if text.endswith(line_ending) != candidate_text.endswith(line_ending):
            _fail("patch proposals must preserve final newline state")
        rendered_diff = _render_unified_diff(
            path=path_value,
            original=text,
            candidate=candidate_text,
        )
        if not rendered_diff:
            _fail("accepted patch changes must produce a nonempty diff")
        diffs.append(rendered_diff)

    if total_original_lines > MAX_PATCH_ORIGINAL_LINES:
        _fail("patch proposal exceeds the changed-line limit")
    unified_diff = "".join(diffs)
    if len(unified_diff) > MAX_PATCH_DIFF_CHARS:
        _fail("generated patch preview exceeds the diff-character limit")
    edits_tuple = tuple(
        sorted(validated, key=lambda edit: (edit.path, edit.start_line, edit.end_line))
    )
    file_snapshots_tuple = tuple(sorted(file_snapshots, key=lambda item: item.path))
    status = PatchProposalValidationStatus.STRUCTURALLY_VALIDATED_UNAPPLIED
    proposal_digest = compute_proposal_digest(
        task_id=task.task_id,
        task_fingerprint=reproduction_result.task_fingerprint,
        reproduction_expectation_fingerprint=(
            reproduction_result.reproduction_expectation_fingerprint
        ),
        hypothesis_id=draft.hypothesis_id,
        model_summary=draft.model_summary,
        validation_status=status,
        validation_summary=PATCH_VALIDATION_SUMMARY,
        edits=edits_tuple,
        file_snapshots=file_snapshots_tuple,
        unified_diff=unified_diff,
    )
    proposal = ValidatedPatchProposal(
        task_id=task.task_id,
        task_fingerprint=reproduction_result.task_fingerprint,
        reproduction_expectation_fingerprint=(
            reproduction_result.reproduction_expectation_fingerprint
        ),
        hypothesis_id=draft.hypothesis_id,
        model_summary=draft.model_summary,
        validation_status=status,
        validation_summary=PATCH_VALIDATION_SUMMARY,
        edits=edits_tuple,
        file_snapshots=file_snapshots_tuple,
        unified_diff=unified_diff,
        proposal_digest=proposal_digest,
    )
    for snapshot in snapshots:
        _verify_snapshot(workspace=workspace, snapshot=snapshot)
    return proposal
