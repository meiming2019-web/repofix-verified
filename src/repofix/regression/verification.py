"""Deterministic post-patch regression command verification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Never

from pydantic import ValidationError

from repofix.agent.reproduction_loop import (
    ReproductionAgentRunResult,
    compute_reproduction_run_fingerprint,
    compute_task_fingerprint,
)
from repofix.agent.state import AgentPhase
from repofix.patching.models import (
    PatchApplicationResult,
    PatchApplicationStatus,
    ValidatedPatchProposal,
    compute_patch_application_result_fingerprint,
)
from repofix.regression._workspace import (
    capture_applied_targets,
    resolve_workspace,
    verify_targets_unchanged,
)
from repofix.regression.models import (
    RegressionBaselineResult,
    RegressionBaselineStatus,
    RegressionVerificationResult,
    compute_regression_baseline_fingerprint,
    compute_regression_specification_fingerprint,
    verification_status_for,
    verification_summary_for,
)
from repofix.reproduction.models import (
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionStatus,
    compute_reproduction_expectation_fingerprint,
)
from repofix.reproduction.post_patch import (
    PostPatchReproductionResult,
    PostPatchReproductionStatus,
    compute_post_patch_reproduction_fingerprint,
)
from repofix.tasks.spec import AgentTaskSpec, RegressionSpecification

if TYPE_CHECKING:
    from repofix.agent import ApprovedCommandGateway


class RegressionVerificationError(RuntimeError):
    """Raised when post-patch regression verification cannot validly run."""


def _fail(message: str, cause: BaseException | None = None) -> Never:
    if cause is None:
        raise RegressionVerificationError(message)
    raise RegressionVerificationError(message) from cause


def _canonical(value: object, model: type[object], description: str) -> object:
    try:
        return model.model_validate(value.model_dump())  # type: ignore[attr-defined]
    except (AttributeError, ValidationError) as error:
        _fail(f"{description} failed canonical integrity checks", error)


def _verify_chain(
    *,
    task: AgentTaskSpec,
    expectation: ReproductionExpectation,
    specification: RegressionSpecification,
    original: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    baseline: RegressionBaselineResult,
    application: PatchApplicationResult,
    post_patch: PostPatchReproductionResult,
) -> tuple[str, str, str, str, str, str, str]:
    task_fingerprint = compute_task_fingerprint(task)
    expectation_fingerprint = compute_reproduction_expectation_fingerprint(expectation)
    run_fingerprint = compute_reproduction_run_fingerprint(original)
    specification_fingerprint = compute_regression_specification_fingerprint(specification)
    baseline_fingerprint = compute_regression_baseline_fingerprint(baseline)
    application_fingerprint = compute_patch_application_result_fingerprint(application)
    post_patch_fingerprint = compute_post_patch_reproduction_fingerprint(post_patch)

    if any(
        task_id != task.task_id
        for task_id in (
            original.state.task_id,
            proposal.task_id,
            baseline.task_id,
            application.task_id,
            post_patch.task_id,
        )
    ):
        _fail("regression verification task identity does not match")
    if any(
        value != task_fingerprint
        for value in (
            original.task_fingerprint,
            proposal.task_fingerprint,
            baseline.task_fingerprint,
            application.task_fingerprint,
            post_patch.task_fingerprint,
        )
    ):
        _fail("regression verification task fingerprint does not match")
    if any(
        value != expectation_fingerprint
        for value in (
            original.reproduction_expectation_fingerprint,
            proposal.reproduction_expectation_fingerprint,
            baseline.reproduction_expectation_fingerprint,
            application.reproduction_expectation_fingerprint,
            post_patch.reproduction_expectation_fingerprint,
        )
    ):
        _fail("regression verification expectation fingerprint does not match")
    if any(
        value != run_fingerprint
        for value in (
            proposal.reproduction_run_fingerprint,
            baseline.original_reproduction_run_fingerprint,
            application.reproduction_run_fingerprint,
            post_patch.original_reproduction_run_fingerprint,
        )
    ):
        _fail("regression verification reproduction run fingerprint does not match")
    if any(
        value != proposal.proposal_digest
        for value in (
            baseline.proposal_digest,
            application.proposal_digest,
            post_patch.proposal_digest,
        )
    ):
        _fail("regression verification proposal digest does not match")
    if baseline.regression_specification_fingerprint != specification_fingerprint:
        _fail("regression verification specification fingerprint does not match")
    if baseline.command_id != specification.command_id:
        _fail("regression baseline command does not match the current specification")
    if specification.command_id not in task.approved_commands:
        _fail("regression verification command is not approved by the current task")
    if baseline.evidence.argv != task.approved_commands[specification.command_id].argv:
        _fail("regression baseline arguments do not match the current task")
    if baseline.status is not RegressionBaselineStatus.PASSED:
        _fail("post-patch regression verification requires a passing baseline")
    if (
        original.state.phase is not AgentPhase.FINISHED
        or len(original.attempts) != 1
        or original.attempts[0].verdict.status is not ReproductionStatus.REPRODUCED
    ):
        _fail("regression verification requires completed original reproduction")
    if application.status is not PatchApplicationStatus.APPLIED:
        _fail("regression verification requires an applied patch result")
    if post_patch.application_status is not application.status:
        _fail("post-patch reproduction does not match the application status")
    if post_patch.command_id != expectation.command_id:
        _fail("post-patch reproduction command does not match the expectation")
    if post_patch.evidence.argv != task.approved_commands[expectation.command_id].argv:
        _fail("post-patch reproduction arguments do not match the current task")
    if (
        post_patch.status
        is not PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
    ):
        _fail("original behavior must be absent before regression verification")
    return (
        task_fingerprint,
        expectation_fingerprint,
        run_fingerprint,
        specification_fingerprint,
        baseline_fingerprint,
        application_fingerprint,
        post_patch_fingerprint,
    )


def verify_post_patch_regression(
    *,
    workspace_root: Path,
    task: AgentTaskSpec,
    reproduction_expectation: ReproductionExpectation,
    regression_specification: RegressionSpecification,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    baseline_result: RegressionBaselineResult,
    application_result: PatchApplicationResult,
    post_patch_reproduction_result: PostPatchReproductionResult,
    command_gateway: ApprovedCommandGateway,
) -> RegressionVerificationResult:
    """Execute the passing baseline's command once after the reproduction gate."""
    canonical_task = _canonical(task, AgentTaskSpec, "current task")
    canonical_expectation = _canonical(
        reproduction_expectation,
        ReproductionExpectation,
        "reproduction expectation",
    )
    canonical_specification = _canonical(
        regression_specification,
        RegressionSpecification,
        "regression specification",
    )
    canonical_original = _canonical(
        original_reproduction_result,
        ReproductionAgentRunResult,
        "original reproduction result",
    )
    canonical_proposal = _canonical(
        proposal,
        ValidatedPatchProposal,
        "validated proposal",
    )
    canonical_baseline = _canonical(
        baseline_result,
        RegressionBaselineResult,
        "regression baseline",
    )
    canonical_application = _canonical(
        application_result,
        PatchApplicationResult,
        "patch application result",
    )
    canonical_post_patch = _canonical(
        post_patch_reproduction_result,
        PostPatchReproductionResult,
        "post-patch reproduction result",
    )
    assert isinstance(canonical_task, AgentTaskSpec)
    assert isinstance(canonical_expectation, ReproductionExpectation)
    assert isinstance(canonical_specification, RegressionSpecification)
    assert isinstance(canonical_original, ReproductionAgentRunResult)
    assert isinstance(canonical_proposal, ValidatedPatchProposal)
    assert isinstance(canonical_baseline, RegressionBaselineResult)
    assert isinstance(canonical_application, PatchApplicationResult)
    assert isinstance(canonical_post_patch, PostPatchReproductionResult)
    (
        task_fingerprint,
        expectation_fingerprint,
        run_fingerprint,
        specification_fingerprint,
        baseline_fingerprint,
        application_fingerprint,
        post_patch_fingerprint,
    ) = _verify_chain(
        task=canonical_task,
        expectation=canonical_expectation,
        specification=canonical_specification,
        original=canonical_original,
        proposal=canonical_proposal,
        baseline=canonical_baseline,
        application=canonical_application,
        post_patch=canonical_post_patch,
    )
    workspace = resolve_workspace(
        workspace_root,
        error_type=RegressionVerificationError,
        stage="regression verification",
    )
    before = capture_applied_targets(
        workspace=workspace,
        proposal=canonical_proposal,
        application=canonical_application,
        error_factory=RegressionVerificationError,
        stage="regression verification",
    )

    command_id = canonical_specification.command_id
    execution_result = command_gateway.execute(command_id)
    if execution_result.command_id != command_id:
        _fail("regression command gateway returned an inconsistent command ID")
    if execution_result.argv != canonical_task.approved_commands[command_id].argv:
        _fail("regression command gateway returned inconsistent arguments")
    verify_targets_unchanged(
        workspace=workspace,
        before=before,
        error_factory=RegressionVerificationError,
        stage="regression verification",
    )
    evidence = ReproductionEvidence.from_execution_result(execution_result)
    status = verification_status_for(evidence)
    return RegressionVerificationResult(
        task_id=canonical_task.task_id,
        task_fingerprint=task_fingerprint,
        regression_specification_fingerprint=specification_fingerprint,
        regression_baseline_fingerprint=baseline_fingerprint,
        reproduction_expectation_fingerprint=expectation_fingerprint,
        original_reproduction_run_fingerprint=run_fingerprint,
        proposal_digest=canonical_proposal.proposal_digest,
        application_result_fingerprint=application_fingerprint,
        post_patch_reproduction_fingerprint=post_patch_fingerprint,
        application_status=canonical_application.status,
        post_patch_reproduction_status=canonical_post_patch.status,
        command_id=command_id,
        evidence=evidence,
        status=status,
        verification_summary=verification_summary_for(status),
    )
