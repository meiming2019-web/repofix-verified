"""Proposal-target integrity helpers shared by regression stages."""

from pathlib import Path, PurePosixPath
from typing import Callable, TypeVar

from repofix.patching.models import PatchApplicationResult, ValidatedPatchProposal
from repofix.patching.validator import (
    PatchProposalValidationError,
    _FileSnapshot,
    _read_snapshot,
    _verify_snapshot,
)


ErrorT = TypeVar("ErrorT", bound=RuntimeError)


def resolve_workspace(
    workspace_root: Path,
    *,
    error_type: type[ErrorT],
    stage: str,
) -> Path:
    try:
        workspace = workspace_root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise error_type(f"{stage} workspace could not be resolved") from error
    if not workspace.is_dir():
        raise error_type(f"{stage} workspace is not a directory")
    return workspace


def read_target(
    *,
    workspace: Path,
    path: str,
    error_factory: Callable[[str], ErrorT],
    stage: str,
) -> _FileSnapshot:
    try:
        return _read_snapshot(workspace=workspace, logical=PurePosixPath(path))
    except PatchProposalValidationError as error:
        raise error_factory(f"{stage} target could not be safely inspected") from error


def capture_original_targets(
    *,
    workspace: Path,
    proposal: ValidatedPatchProposal,
    error_factory: Callable[[str], ErrorT],
    stage: str,
) -> dict[str, _FileSnapshot]:
    snapshots: dict[str, _FileSnapshot] = {}
    for expected in proposal.file_snapshots:
        current = read_target(
            workspace=workspace,
            path=expected.path,
            error_factory=error_factory,
            stage=stage,
        )
        if (
            current.sha256 != expected.original_file_sha256
            or current.size != expected.size_bytes
        ):
            raise error_factory(f"{stage} workspace does not match proposal originals")
        snapshots[expected.path] = current
    return snapshots


def verify_targets_unchanged(
    *,
    workspace: Path,
    before: dict[str, _FileSnapshot],
    error_factory: Callable[[str], ErrorT],
    stage: str,
) -> None:
    for expected in before.values():
        try:
            _verify_snapshot(workspace=workspace, snapshot=expected)
        except PatchProposalValidationError as error:
            raise error_factory(f"{stage} command modified a proposal target") from error


def capture_applied_targets(
    *,
    workspace: Path,
    proposal: ValidatedPatchProposal,
    application: PatchApplicationResult,
    error_factory: Callable[[str], ErrorT],
    stage: str,
) -> dict[str, _FileSnapshot]:
    """Validate applied candidate metadata and capture current proposal targets."""
    proposals = {item.path: item for item in proposal.file_snapshots}
    applied = {item.path: item for item in application.files}
    if set(proposals) != set(applied):
        raise error_factory(f"{stage} application paths do not match the proposal")
    before: dict[str, _FileSnapshot] = {}
    for path in sorted(proposals):
        expected = proposals[path]
        actual = applied[path]
        if (
            actual.original_file_sha256 != expected.original_file_sha256
            or actual.original_size_bytes != expected.size_bytes
            or actual.candidate_file_sha256 != expected.candidate_file_sha256
            or actual.candidate_size_bytes != expected.candidate_size_bytes
        ):
            raise error_factory(f"{stage} application metadata does not match the proposal")
        current = read_target(
            workspace=workspace,
            path=path,
            error_factory=error_factory,
            stage=stage,
        )
        if (
            current.sha256 != actual.candidate_file_sha256
            or current.size != actual.candidate_size_bytes
        ):
            raise error_factory(f"{stage} workspace does not match the applied candidate")
        before[path] = current
    return before
