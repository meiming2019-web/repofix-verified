"""Safe resolution of one evaluator-owned hidden test asset."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from repofix.patching.validator import (
    PatchProposalValidationError,
    _FileSnapshot,
    _read_snapshot,
    _verify_snapshot,
)
from repofix.tasks.spec import (
    ApprovedCommand,
    HiddenVerificationSpecification,
)

from repofix.hidden.errors import HiddenVerificationError


@dataclass(frozen=True)
class ResolvedHiddenCommand:
    """Trusted runtime command and pre-execution hidden-file snapshot."""

    workspace: Path
    evaluator_assets_root: Path
    command_id: str
    command: ApprovedCommand
    hidden_file: _FileSnapshot


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_hidden_command(
    *,
    workspace_root: Path,
    evaluator_assets_root: Path,
    specification: HiddenVerificationSpecification,
) -> ResolvedHiddenCommand:
    """Resolve and validate the singleton hidden command without executing it."""
    try:
        workspace = workspace_root.resolve(strict=True)
        assets_root = evaluator_assets_root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise HiddenVerificationError(
            "hidden verification roots could not be resolved"
        ) from error
    if not workspace.is_dir() or not assets_root.is_dir():
        raise HiddenVerificationError("hidden verification roots must be directories")
    if _contains(workspace, assets_root):
        raise HiddenVerificationError(
            "evaluator assets root must be outside the agent workspace"
        )

    try:
        snapshot = _read_snapshot(
            workspace=assets_root,
            logical=PurePosixPath(specification.test_file.path),
        )
    except PatchProposalValidationError as error:
        raise HiddenVerificationError(
            "hidden evaluator asset could not be safely inspected"
        ) from error
    if _contains(workspace, snapshot.resolved):
        raise HiddenVerificationError(
            "hidden evaluator asset must be outside the agent workspace"
        )
    if (
        snapshot.sha256 != specification.test_file.sha256
        or snapshot.size != specification.test_file.size_bytes
    ):
        raise HiddenVerificationError(
            "hidden evaluator asset does not match its specification"
        )
    command = ApprovedCommand(
        argv=(*specification.launcher.argv, str(snapshot.resolved))
    )
    return ResolvedHiddenCommand(
        workspace=workspace,
        evaluator_assets_root=assets_root,
        command_id=specification.command_id,
        command=command,
        hidden_file=snapshot,
    )


def verify_hidden_file_unchanged(resolution: ResolvedHiddenCommand) -> None:
    """Require the evaluator-only file to retain its full safe snapshot identity."""
    try:
        _verify_snapshot(
            workspace=resolution.evaluator_assets_root,
            snapshot=resolution.hidden_file,
        )
    except PatchProposalValidationError as error:
        raise HiddenVerificationError(
            "hidden command modified the evaluator-owned test file"
        ) from error
