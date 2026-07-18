"""Independent verification of the original reproduction after patch application."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Never, Self

from pydantic import field_validator, model_validator, ValidationError

from repofix.execution import ApprovedCommandExecutionResult
from repofix.patching.models import (
    PatchApplicationResult,
    PatchApplicationStatus,
    ValidatedPatchProposal,
)
from repofix.reproduction.models import (
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionStatus,
    ReproductionVerdict,
    compute_reproduction_expectation_fingerprint,
)
from repofix.reproduction.verifier import verify_reproduction
from repofix.tasks import AgentTaskSpec
from repofix.tasks.spec import StrictFrozenModel

if TYPE_CHECKING:
    from repofix.agent import ApprovedCommandGateway, ReproductionAgentRunResult
    from repofix.patching.validator import _FileSnapshot


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PostPatchReproductionError(RuntimeError):
    """Raised when post-patch reproduction cannot be validly performed."""


class PostPatchReproductionStatus(StrEnum):
    ORIGINAL_BEHAVIOR_NOT_REPRODUCED = "original_behavior_not_reproduced"
    ORIGINAL_BEHAVIOR_STILL_REPRODUCED = "original_behavior_still_reproduced"
    INCONCLUSIVE = "inconclusive"


POST_PATCH_NOT_REPRODUCED_SUMMARY = (
    "The original reproduced behavior was not observed after the patch was applied. "
    "Regression and hidden verification have not run."
)
POST_PATCH_STILL_REPRODUCED_SUMMARY = (
    "The original reproduced behavior is still present after the patch was applied. "
    "Regression and hidden verification have not run."
)
POST_PATCH_INCONCLUSIVE_SUMMARY = (
    "Post-patch reproduction verification was inconclusive. "
    "Regression and hidden verification have not run."
)


_STATUS_BY_VERDICT = {
    ReproductionStatus.NOT_REPRODUCED: (
        PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
    ),
    ReproductionStatus.REPRODUCED: (
        PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_STILL_REPRODUCED
    ),
    ReproductionStatus.INCONCLUSIVE: PostPatchReproductionStatus.INCONCLUSIVE,
}
_SUMMARY_BY_STATUS = {
    PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED: (
        POST_PATCH_NOT_REPRODUCED_SUMMARY
    ),
    PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_STILL_REPRODUCED: (
        POST_PATCH_STILL_REPRODUCED_SUMMARY
    ),
    PostPatchReproductionStatus.INCONCLUSIVE: POST_PATCH_INCONCLUSIVE_SUMMARY,
}


class PostPatchReproductionResult(StrictFrozenModel):
    task_id: str
    task_fingerprint: str
    reproduction_expectation_fingerprint: str
    original_reproduction_run_fingerprint: str
    proposal_digest: str
    application_status: PatchApplicationStatus
    status: PostPatchReproductionStatus
    command_id: str
    evidence: ReproductionEvidence
    verifier_verdict: ReproductionVerdict
    verification_summary: str

    @field_validator(
        "task_fingerprint",
        "reproduction_expectation_fingerprint",
        "original_reproduction_run_fingerprint",
        "proposal_digest",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("post-patch hashes must be lowercase hexadecimal SHA-256")
        return value

    @model_validator(mode="after")
    def validate_canonical_result(self) -> Self:
        if self.application_status is not PatchApplicationStatus.APPLIED:
            raise ValueError("post-patch verification requires applied status")
        if (
            self.command_id != self.evidence.command_id
            or self.command_id != self.verifier_verdict.command_id
            or self.evidence.exit_code != self.verifier_verdict.exit_code
        ):
            raise ValueError("post-patch command evidence and verdict identity must match")
        if self.status is not _STATUS_BY_VERDICT[self.verifier_verdict.status]:
            raise ValueError("post-patch status does not match the verifier verdict")
        if self.verification_summary != _SUMMARY_BY_STATUS[self.status]:
            raise ValueError("post-patch result requires its canonical system summary")
        return self


def _fail(message: str, cause: BaseException | None = None) -> Never:
    if cause is None:
        raise PostPatchReproductionError(message)
    raise PostPatchReproductionError(message) from cause


def _canonical_inputs(
    *,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    application_result: PatchApplicationResult,
) -> tuple[ReproductionAgentRunResult, ValidatedPatchProposal, PatchApplicationResult]:
    from repofix.agent.reproduction_loop import ReproductionAgentRunResult

    if application_result.status is not PatchApplicationStatus.APPLIED:
        _fail("post-patch verification requires an applied patch result")
    try:
        reproduction_result = ReproductionAgentRunResult.model_validate(
            original_reproduction_result.model_dump()
        )
    except ValidationError as error:
        _fail("original reproduction result failed canonical integrity checks", error)
    try:
        canonical_proposal = ValidatedPatchProposal.model_validate(proposal.model_dump())
    except ValidationError as error:
        _fail("proposal digest or canonical integrity checks failed", error)
    try:
        canonical_application = PatchApplicationResult.model_validate(
            application_result.model_dump()
        )
    except ValidationError as error:
        _fail("patch application result failed canonical integrity checks", error)
    return reproduction_result, canonical_proposal, canonical_application


def _verify_bindings(
    *,
    task: AgentTaskSpec,
    expectation: ReproductionExpectation,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    application_result: PatchApplicationResult,
) -> None:
    from repofix.agent.reproduction_loop import (
        compute_reproduction_run_fingerprint,
        compute_task_fingerprint,
    )
    from repofix.agent.state import AgentPhase

    task_fingerprint = compute_task_fingerprint(task)
    expectation_fingerprint = compute_reproduction_expectation_fingerprint(expectation)
    reproduction_run_fingerprint = compute_reproduction_run_fingerprint(
        original_reproduction_result
    )
    if (
        original_reproduction_result.state.task_id != task.task_id
        or proposal.task_id != task.task_id
        or application_result.task_id != task.task_id
    ):
        _fail("post-patch verification task identity does not match")
    if (
        original_reproduction_result.task_fingerprint != task_fingerprint
        or proposal.task_fingerprint != task_fingerprint
        or application_result.task_fingerprint != task_fingerprint
    ):
        _fail("post-patch verification task fingerprint does not match")
    if (
        original_reproduction_result.reproduction_expectation_fingerprint
        != expectation_fingerprint
        or proposal.reproduction_expectation_fingerprint != expectation_fingerprint
        or application_result.reproduction_expectation_fingerprint
        != expectation_fingerprint
    ):
        _fail("post-patch verification expectation fingerprint does not match")
    if (
        proposal.reproduction_run_fingerprint != reproduction_run_fingerprint
        or application_result.reproduction_run_fingerprint
        != reproduction_run_fingerprint
    ):
        _fail("post-patch verification reproduction run fingerprint does not match")
    if application_result.proposal_digest != proposal.proposal_digest:
        _fail("post-patch verification proposal digest does not match")
    if application_result.status is not PatchApplicationStatus.APPLIED:
        _fail("post-patch verification requires an applied patch result")
    if (
        original_reproduction_result.state.phase is not AgentPhase.FINISHED
        or len(original_reproduction_result.attempts) != 1
        or original_reproduction_result.attempts[0].verdict.status
        is not ReproductionStatus.REPRODUCED
    ):
        _fail("post-patch verification requires completed original reproduction")
    if expectation.command_id not in task.approved_commands:
        _fail("post-patch reproduction command is not approved by the current task")


def _read_target_snapshot(*, workspace: Path, path: str) -> _FileSnapshot:
    from repofix.patching.validator import PatchProposalValidationError, _read_snapshot

    try:
        return _read_snapshot(workspace=workspace, logical=PurePosixPath(path))
    except PatchProposalValidationError as error:
        _fail("post-patch verification target could not be safely inspected", error)


def _preflight_workspace(
    *,
    workspace: Path,
    proposal: ValidatedPatchProposal,
    application_result: PatchApplicationResult,
) -> dict[str, _FileSnapshot]:
    proposal_snapshots = {snapshot.path: snapshot for snapshot in proposal.file_snapshots}
    applied_files = {item.path: item for item in application_result.files}
    if set(proposal_snapshots) != set(applied_files):
        _fail("post-patch application file paths do not match the proposal")
    before: dict[str, _FileSnapshot] = {}
    for path in sorted(proposal_snapshots):
        expected = proposal_snapshots[path]
        applied = applied_files[path]
        if (
            applied.original_file_sha256 != expected.original_file_sha256
            or applied.original_size_bytes != expected.size_bytes
            or applied.candidate_file_sha256 != expected.candidate_file_sha256
            or applied.candidate_size_bytes != expected.candidate_size_bytes
        ):
            _fail("post-patch application file metadata does not match the proposal")
        current = _read_target_snapshot(workspace=workspace, path=path)
        if (
            current.sha256 != applied.candidate_file_sha256
            or current.size != applied.candidate_size_bytes
        ):
            _fail("post-patch workspace does not match the applied candidate")
        before[path] = current
    return before


def _verify_targets_unchanged(
    *, workspace: Path, before: dict[str, _FileSnapshot]
) -> None:
    for path, snapshot in before.items():
        current = _read_target_snapshot(workspace=workspace, path=path)
        if current.sha256 != snapshot.sha256 or current.size != snapshot.size:
            _fail("post-patch command modified an applied proposal target")


def _execute_once(
    *,
    task: AgentTaskSpec,
    expectation: ReproductionExpectation,
    command_gateway: ApprovedCommandGateway,
) -> ApprovedCommandExecutionResult:
    result = command_gateway.execute(expectation.command_id)
    approved_argv = task.approved_commands[expectation.command_id].argv
    if result.command_id != expectation.command_id:
        _fail("post-patch command gateway returned an inconsistent command ID")
    if result.argv != approved_argv:
        _fail("post-patch command gateway returned inconsistent command arguments")
    return result


def verify_post_patch_reproduction(
    *,
    workspace_root: Path,
    task: AgentTaskSpec,
    expectation: ReproductionExpectation,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    application_result: PatchApplicationResult,
    command_gateway: ApprovedCommandGateway,
) -> PostPatchReproductionResult:
    """Rerun and reinterpret the original reproduction after controlled application."""
    reproduction_result, canonical_proposal, canonical_application = _canonical_inputs(
        original_reproduction_result=original_reproduction_result,
        proposal=proposal,
        application_result=application_result,
    )
    _verify_bindings(
        task=task,
        expectation=expectation,
        original_reproduction_result=reproduction_result,
        proposal=canonical_proposal,
        application_result=canonical_application,
    )
    try:
        workspace = workspace_root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        _fail("post-patch verification workspace could not be resolved", error)
    if not workspace.is_dir():
        _fail("post-patch verification workspace is not a directory")
    before = _preflight_workspace(
        workspace=workspace,
        proposal=canonical_proposal,
        application_result=canonical_application,
    )

    execution_result = _execute_once(
        task=task,
        expectation=expectation,
        command_gateway=command_gateway,
    )
    _verify_targets_unchanged(workspace=workspace, before=before)
    evidence = ReproductionEvidence.from_execution_result(execution_result)
    verdict = verify_reproduction(expectation=expectation, evidence=evidence)
    status = _STATUS_BY_VERDICT[verdict.status]
    return PostPatchReproductionResult(
        task_id=task.task_id,
        task_fingerprint=canonical_proposal.task_fingerprint,
        reproduction_expectation_fingerprint=(
            canonical_proposal.reproduction_expectation_fingerprint
        ),
        original_reproduction_run_fingerprint=(
            canonical_proposal.reproduction_run_fingerprint
        ),
        proposal_digest=canonical_proposal.proposal_digest,
        application_status=canonical_application.status,
        status=status,
        command_id=expectation.command_id,
        evidence=evidence,
        verifier_verdict=verdict,
        verification_summary=_SUMMARY_BY_STATUS[status],
    )
