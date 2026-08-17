"""Proposal-target integrity helpers shared by regression stages."""

from pathlib import Path, PurePosixPath
from typing import Callable, TypeVar

from repofix.patching.models import ValidatedPatchProposal
from repofix.patching.validator import (
    PatchProposalValidationError,
    _FileSnapshot,
    _read_snapshot,
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
    for path, expected in before.items():
        current = read_target(
            workspace=workspace,
            path=path,
            error_factory=error_factory,
            stage=stage,
        )
        if current.sha256 != expected.sha256 or current.size != expected.size:
            raise error_factory(f"{stage} command modified a proposal target")
