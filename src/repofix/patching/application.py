"""Controlled application of one validated patch proposal."""

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Never

from pydantic import ValidationError

from repofix.agent import AgentPhase, ReproductionAgentRunResult
from repofix.agent.reproduction_loop import (
    compute_reproduction_run_fingerprint,
    compute_task_fingerprint,
)
from repofix.patching.models import (
    PATCH_APPLICATION_SUMMARY,
    AppliedPatchFile,
    PatchApplicationResult,
    PatchApplicationStatus,
    ValidatedPatchEdit,
    ValidatedPatchFileSnapshot,
    ValidatedPatchProposal,
)
from repofix.patching.validator import (
    PatchProposalValidationError,
    _FileSnapshot,
    _read_snapshot,
    reconstruct_candidate_bytes,
)
from repofix.reproduction import ReproductionStatus
from repofix.tasks import AgentTaskSpec


class PatchApplicationError(RuntimeError):
    """Raised when a validated patch proposal cannot be safely applied."""


@dataclass
class _PreflightFile:
    path: str
    target: Path
    snapshot: _FileSnapshot
    proposal_snapshot: ValidatedPatchFileSnapshot
    candidate: bytes


@dataclass
class _PreparedFile:
    preflight: _PreflightFile
    temporary: Path | None


def _fail(message: str, cause: BaseException | None = None) -> Never:
    if cause is None:
        raise PatchApplicationError(message)
    raise PatchApplicationError(message) from cause


def _canonical_proposal(proposal: ValidatedPatchProposal) -> ValidatedPatchProposal:
    try:
        return ValidatedPatchProposal.model_validate(proposal.model_dump())
    except ValidationError as error:
        _fail("validated patch proposal failed its canonical integrity checks", error)


def _canonical_reproduction_result(
    result: ReproductionAgentRunResult,
) -> ReproductionAgentRunResult:
    try:
        return ReproductionAgentRunResult.model_validate(result.model_dump())
    except ValidationError as error:
        _fail("reproduction result failed its canonical integrity checks", error)


def _verify_bindings(
    *,
    task: AgentTaskSpec,
    reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
) -> None:
    task_fingerprint = compute_task_fingerprint(task)
    run_fingerprint = compute_reproduction_run_fingerprint(reproduction_result)
    if proposal.task_id != task.task_id or reproduction_result.state.task_id != task.task_id:
        _fail("patch application task identity does not match")
    if (
        proposal.task_fingerprint != task_fingerprint
        or reproduction_result.task_fingerprint != task_fingerprint
    ):
        _fail("patch application task fingerprint does not match")
    if (
        proposal.reproduction_expectation_fingerprint
        != reproduction_result.reproduction_expectation_fingerprint
    ):
        _fail("patch application reproduction expectation fingerprint does not match")
    if proposal.reproduction_run_fingerprint != run_fingerprint:
        _fail("patch application reproduction run fingerprint does not match")
    if (
        reproduction_result.state.phase is not AgentPhase.FINISHED
        or len(reproduction_result.attempts) != 1
        or reproduction_result.attempts[0].verdict.status is not ReproductionStatus.REPRODUCED
    ):
        _fail("patch application requires completed verified reproduction")


def _snapshot_for_application(*, workspace: Path, logical: PurePosixPath) -> _FileSnapshot:
    try:
        return _read_snapshot(workspace=workspace, logical=logical)
    except PatchProposalValidationError as error:
        _fail("patch application target could not be safely inspected", error)


def _candidate_for_application(
    *, original: bytes, edits: tuple[ValidatedPatchEdit, ...]
) -> bytes:
    try:
        return reconstruct_candidate_bytes(original_bytes=original, edits=edits)
    except PatchProposalValidationError as error:
        _fail("patch application candidate could not be reconstructed", error)


def _preflight_files(
    *, workspace: Path, task: AgentTaskSpec, proposal: ValidatedPatchProposal
) -> list[_PreflightFile]:
    roots = tuple(PurePosixPath(value) for value in task.patchable_source_paths)
    snapshots = {snapshot.path: snapshot for snapshot in proposal.file_snapshots}
    edits_by_path: dict[str, list[ValidatedPatchEdit]] = {}
    for edit in proposal.edits:
        edits_by_path.setdefault(edit.path, []).append(edit)
    if set(snapshots) != set(edits_by_path):
        _fail("patch application requires exactly one snapshot for each target")

    identities: dict[tuple[int, int], str] = {}
    preflight: list[_PreflightFile] = []
    for path in sorted(edits_by_path):
        logical = PurePosixPath(path)
        if not any(logical == root or logical.is_relative_to(root) for root in roots):
            _fail("patch application target is outside patchable source paths")
        current = _snapshot_for_application(workspace=workspace, logical=logical)
        identity = (current.device, current.inode)
        previous = identities.setdefault(identity, path)
        if previous != path:
            _fail("multiple patch application paths refer to the same physical file")
        expected = snapshots[path]
        if current.sha256 != expected.original_file_sha256:
            _fail("patch application target hash does not match the proposal")
        if current.size != expected.size_bytes:
            _fail("patch application target size does not match the proposal")
        candidate = _candidate_for_application(
            original=current.contents,
            edits=tuple(edits_by_path[path]),
        )
        if hashlib.sha256(candidate).hexdigest() != expected.candidate_file_sha256:
            _fail("reconstructed patch candidate hash does not match the proposal")
        if len(candidate) != expected.candidate_size_bytes:
            _fail("reconstructed patch candidate size does not match the proposal")
        preflight.append(
            _PreflightFile(
                path=path,
                target=workspace.joinpath(*logical.parts),
                snapshot=current,
                proposal_snapshot=expected,
                candidate=candidate,
            )
        )
    return preflight


def _prepare_temporary(target: Path, contents: bytes, mode: int) -> Path:
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, temporary_value = tempfile.mkstemp(
            prefix=".repofix-patch-",
            dir=target.parent,
        )
        temporary = Path(temporary_value)
        os.fchmod(descriptor, stat.S_IMODE(mode))
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except OSError as error:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        _fail("patch application temporary file could not be prepared", error)


def _cleanup_temporaries(prepared: list[_PreparedFile]) -> OSError | None:
    first_error: OSError | None = None
    for item in prepared:
        if item.temporary is None:
            continue
        try:
            item.temporary.unlink(missing_ok=True)
        except OSError as error:
            if first_error is None:
                first_error = error
    return first_error


def _rollback_replaced(replaced: list[_PreparedFile]) -> bool:
    failed = False
    for item in reversed(replaced):
        rollback_temporary: Path | None = None
        try:
            rollback_temporary = _prepare_temporary(
                item.preflight.target,
                item.preflight.snapshot.contents,
                item.preflight.snapshot.mode,
            )
            os.replace(rollback_temporary, item.preflight.target)
            rollback_temporary = None
        except (OSError, PatchApplicationError):
            failed = True
        finally:
            if rollback_temporary is not None:
                try:
                    rollback_temporary.unlink(missing_ok=True)
                except OSError:
                    failed = True
    return failed


def _write_preflighted_files(preflight: list[_PreflightFile]) -> None:
    prepared: list[_PreparedFile] = []
    try:
        for preflight_item in preflight:
            temporary = _prepare_temporary(
                preflight_item.target,
                preflight_item.candidate,
                preflight_item.snapshot.mode,
            )
            prepared.append(_PreparedFile(preflight=preflight_item, temporary=temporary))
    except PatchApplicationError:
        cleanup_error = _cleanup_temporaries(prepared)
        if cleanup_error is not None:
            _fail("patch application failed and temporary files may require manual cleanup")
        raise

    replaced: list[_PreparedFile] = []
    try:
        for prepared_item in prepared:
            assert prepared_item.temporary is not None
            os.replace(prepared_item.temporary, prepared_item.preflight.target)
            prepared_item.temporary = None
            replaced.append(prepared_item)
    except OSError as error:
        rollback_failed = _rollback_replaced(replaced)
        cleanup_failed = _cleanup_temporaries(prepared) is not None
        if rollback_failed:
            _fail(
                "patch application rollback failed; the workspace may require manual restoration",
                error,
            )
        if cleanup_failed:
            _fail("patch application failed and temporary files may require manual cleanup", error)
        _fail("patch application could not atomically replace all targets", error)

    cleanup_error = _cleanup_temporaries(prepared)
    if cleanup_error is not None:
        _fail("patch application completed but temporary files could not be cleaned", cleanup_error)


def apply_validated_patch_proposal(
    *,
    workspace_root: Path,
    task: AgentTaskSpec,
    reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
) -> PatchApplicationResult:
    """Revalidate and atomically write one bounded proposal to a controlled workspace."""
    canonical_proposal = _canonical_proposal(proposal)
    canonical_result = _canonical_reproduction_result(reproduction_result)
    _verify_bindings(
        task=task,
        reproduction_result=canonical_result,
        proposal=canonical_proposal,
    )
    if not task.patchable_source_paths:
        _fail("patch application task does not configure patchable source paths")
    try:
        workspace = workspace_root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        _fail("patch application workspace could not be resolved", error)
    if not workspace.is_dir():
        _fail("patch application workspace is not a directory")

    preflight = _preflight_files(workspace=workspace, task=task, proposal=canonical_proposal)
    _write_preflighted_files(preflight)
    files = tuple(
        AppliedPatchFile(
            path=item.path,
            original_file_sha256=item.proposal_snapshot.original_file_sha256,
            candidate_file_sha256=item.proposal_snapshot.candidate_file_sha256,
            original_size_bytes=item.proposal_snapshot.size_bytes,
            candidate_size_bytes=item.proposal_snapshot.candidate_size_bytes,
        )
        for item in preflight
    )
    return PatchApplicationResult(
        task_id=task.task_id,
        task_fingerprint=canonical_proposal.task_fingerprint,
        reproduction_expectation_fingerprint=(
            canonical_proposal.reproduction_expectation_fingerprint
        ),
        reproduction_run_fingerprint=canonical_proposal.reproduction_run_fingerprint,
        proposal_digest=canonical_proposal.proposal_digest,
        status=PatchApplicationStatus.APPLIED,
        files=files,
        application_summary=PATCH_APPLICATION_SUMMARY,
    )
