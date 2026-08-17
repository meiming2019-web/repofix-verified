"""Provider-independent evaluator-only hidden verification."""

from __future__ import annotations

from pathlib import Path
from typing import Never, TypeVar

from pydantic import BaseModel, ValidationError

from repofix.agent.reproduction_loop import (
    ReproductionAgentRunResult,
    compute_reproduction_run_fingerprint,
    compute_task_fingerprint,
)
from repofix.agent.state import AgentPhase
from repofix.hidden.assets import (
    ResolvedHiddenCommand,
    verify_hidden_file_unchanged,
)
from repofix.hidden.errors import HiddenVerificationError
from repofix.hidden.interfaces import EvaluatorCommandGateway
from repofix.hidden.models import (
    HiddenVerificationResult,
    compute_hidden_specification_fingerprint,
    hidden_status_for,
    hidden_summary_for,
)
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
    RegressionVerificationStatus,
    compute_regression_baseline_fingerprint,
    compute_regression_specification_fingerprint,
    compute_regression_verification_fingerprint,
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
from repofix.tasks.spec import (
    AgentTaskSpec,
    HiddenVerificationSpecification,
    RegressionSpecification,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _fail(message: str, cause: BaseException | None = None) -> Never:
    if cause is None:
        raise HiddenVerificationError(message)
    raise HiddenVerificationError(message) from cause


def _canonical(value: object, model: type[ModelT], description: str) -> ModelT:
    try:
        if not isinstance(value, model):
            _fail(f"{description} has an invalid type")
        return model.model_validate(value.model_dump())
    except ValidationError as error:
        _fail(f"{description} failed canonical integrity checks", error)


def _verify_chain(
    *,
    task: AgentTaskSpec,
    expectation: ReproductionExpectation,
    regression_specification: RegressionSpecification,
    hidden_specification: HiddenVerificationSpecification,
    original: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    baseline: RegressionBaselineResult,
    application: PatchApplicationResult,
    post_patch: PostPatchReproductionResult,
    regression: RegressionVerificationResult,
) -> tuple[str, str, str, str, str, str, str, str]:
    task_fingerprint = compute_task_fingerprint(task)
    expectation_fingerprint = compute_reproduction_expectation_fingerprint(expectation)
    run_fingerprint = compute_reproduction_run_fingerprint(original)
    regression_specification_fingerprint = (
        compute_regression_specification_fingerprint(regression_specification)
    )
    hidden_specification_fingerprint = compute_hidden_specification_fingerprint(
        hidden_specification
    )
    baseline_fingerprint = compute_regression_baseline_fingerprint(baseline)
    application_fingerprint = compute_patch_application_result_fingerprint(application)
    post_patch_fingerprint = compute_post_patch_reproduction_fingerprint(post_patch)
    regression_fingerprint = compute_regression_verification_fingerprint(regression)

    if hidden_specification.command_id in task.approved_commands:
        _fail("hidden command ID must not be agent-visible")
    if any(
        value != task.task_id
        for value in (
            original.state.task_id,
            proposal.task_id,
            baseline.task_id,
            application.task_id,
            post_patch.task_id,
            regression.task_id,
        )
    ):
        _fail("hidden verification task identity does not match")
    if any(
        value != task_fingerprint
        for value in (
            original.task_fingerprint,
            proposal.task_fingerprint,
            baseline.task_fingerprint,
            application.task_fingerprint,
            post_patch.task_fingerprint,
            regression.task_fingerprint,
        )
    ):
        _fail("hidden verification task fingerprint does not match")
    if any(
        value != expectation_fingerprint
        for value in (
            original.reproduction_expectation_fingerprint,
            proposal.reproduction_expectation_fingerprint,
            baseline.reproduction_expectation_fingerprint,
            application.reproduction_expectation_fingerprint,
            post_patch.reproduction_expectation_fingerprint,
            regression.reproduction_expectation_fingerprint,
        )
    ):
        _fail("hidden verification expectation fingerprint does not match")
    if any(
        value != run_fingerprint
        for value in (
            proposal.reproduction_run_fingerprint,
            baseline.original_reproduction_run_fingerprint,
            application.reproduction_run_fingerprint,
            post_patch.original_reproduction_run_fingerprint,
            regression.original_reproduction_run_fingerprint,
        )
    ):
        _fail("hidden verification reproduction run fingerprint does not match")
    if any(
        value != proposal.proposal_digest
        for value in (
            baseline.proposal_digest,
            application.proposal_digest,
            post_patch.proposal_digest,
            regression.proposal_digest,
        )
    ):
        _fail("hidden verification proposal digest does not match")
    if (
        baseline.regression_specification_fingerprint
        != regression_specification_fingerprint
        or regression.regression_specification_fingerprint
        != regression_specification_fingerprint
    ):
        _fail("hidden verification regression specification does not match")
    if regression.regression_baseline_fingerprint != baseline_fingerprint:
        _fail("hidden verification regression baseline fingerprint does not match")
    if regression.application_result_fingerprint != application_fingerprint:
        _fail("hidden verification application fingerprint does not match")
    if regression.post_patch_reproduction_fingerprint != post_patch_fingerprint:
        _fail("hidden verification post-patch fingerprint does not match")
    if baseline.command_id != regression_specification.command_id:
        _fail("hidden verification baseline command does not match")
    if regression.command_id != regression_specification.command_id:
        _fail("hidden verification public regression command does not match")
    if regression_specification.command_id not in task.approved_commands:
        _fail("public regression command is not approved by the current task")
    approved_regression_argv = task.approved_commands[
        regression_specification.command_id
    ].argv
    if (
        baseline.evidence.argv != approved_regression_argv
        or regression.evidence.argv != approved_regression_argv
    ):
        _fail("hidden verification public regression arguments do not match")
    if (
        original.state.phase is not AgentPhase.FINISHED
        or len(original.attempts) != 1
        or original.attempts[0].verdict.status is not ReproductionStatus.REPRODUCED
    ):
        _fail("hidden verification requires completed original reproduction")
    if baseline.status is not RegressionBaselineStatus.PASSED:
        _fail("hidden verification requires a passing regression baseline")
    if application.status is not PatchApplicationStatus.APPLIED:
        _fail("hidden verification requires an applied patch result")
    if (
        post_patch.application_status is not application.status
        or regression.application_status is not application.status
    ):
        _fail("hidden verification application statuses do not match")
    if post_patch.command_id != expectation.command_id:
        _fail("hidden verification post-patch command does not match")
    if post_patch.evidence.argv != task.approved_commands[expectation.command_id].argv:
        _fail("hidden verification reproduction arguments do not match")
    if (
        post_patch.status
        is not PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
        or regression.post_patch_reproduction_status is not post_patch.status
    ):
        _fail("hidden verification requires the post-patch reproduction gate")
    if (
        regression.status
        is not RegressionVerificationStatus.REGRESSION_COMMAND_PASSED
    ):
        _fail("hidden verification requires the public regression pass gate")
    return (
        task_fingerprint,
        hidden_specification_fingerprint,
        expectation_fingerprint,
        run_fingerprint,
        baseline_fingerprint,
        application_fingerprint,
        post_patch_fingerprint,
        regression_fingerprint,
    )


def _verify_resolution(
    *,
    workspace: Path,
    specification: HiddenVerificationSpecification,
    resolution: ResolvedHiddenCommand,
) -> None:
    if not isinstance(resolution, ResolvedHiddenCommand):
        _fail("hidden command resolution has an invalid type")
    if resolution.workspace != workspace:
        _fail("hidden command resolution does not match the current workspace")
    if resolution.command_id != specification.command_id:
        _fail("hidden command resolution has an inconsistent command ID")
    if resolution.hidden_file.logical.as_posix() != specification.test_file.path:
        _fail("hidden command resolution does not match the evaluator asset")
    if (
        resolution.hidden_file.sha256 != specification.test_file.sha256
        or resolution.hidden_file.size != specification.test_file.size_bytes
    ):
        _fail("hidden command resolution does not match the evaluator specification")
    expected_argv = (
        *specification.launcher.argv,
        str(resolution.hidden_file.resolved),
    )
    if resolution.command.argv != expected_argv:
        _fail("hidden command resolution has inconsistent command arguments")


def verify_hidden_behavior(
    *,
    workspace_root: Path,
    task: AgentTaskSpec,
    reproduction_expectation: ReproductionExpectation,
    regression_specification: RegressionSpecification,
    hidden_specification: HiddenVerificationSpecification,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    regression_baseline_result: RegressionBaselineResult,
    application_result: PatchApplicationResult,
    post_patch_reproduction_result: PostPatchReproductionResult,
    regression_verification_result: RegressionVerificationResult,
    resolved_hidden_command: ResolvedHiddenCommand,
    evaluator_command_gateway: EvaluatorCommandGateway,
) -> HiddenVerificationResult:
    """Execute and classify one evaluator-only hidden command exactly once."""
    canonical_task = _canonical(task, AgentTaskSpec, "current task")
    canonical_expectation = _canonical(
        reproduction_expectation,
        ReproductionExpectation,
        "reproduction expectation",
    )
    canonical_regression_specification = _canonical(
        regression_specification,
        RegressionSpecification,
        "regression specification",
    )
    canonical_hidden_specification = _canonical(
        hidden_specification,
        HiddenVerificationSpecification,
        "hidden specification",
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
        regression_baseline_result,
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
    canonical_regression = _canonical(
        regression_verification_result,
        RegressionVerificationResult,
        "regression verification result",
    )
    (
        task_fingerprint,
        hidden_specification_fingerprint,
        expectation_fingerprint,
        run_fingerprint,
        baseline_fingerprint,
        application_fingerprint,
        post_patch_fingerprint,
        regression_fingerprint,
    ) = _verify_chain(
        task=canonical_task,
        expectation=canonical_expectation,
        regression_specification=canonical_regression_specification,
        hidden_specification=canonical_hidden_specification,
        original=canonical_original,
        proposal=canonical_proposal,
        baseline=canonical_baseline,
        application=canonical_application,
        post_patch=canonical_post_patch,
        regression=canonical_regression,
    )
    workspace = resolve_workspace(
        workspace_root,
        error_type=HiddenVerificationError,
        stage="hidden verification",
    )
    _verify_resolution(
        workspace=workspace,
        specification=canonical_hidden_specification,
        resolution=resolved_hidden_command,
    )
    verify_hidden_file_unchanged(resolved_hidden_command)
    before = capture_applied_targets(
        workspace=workspace,
        proposal=canonical_proposal,
        application=canonical_application,
        error_factory=HiddenVerificationError,
        stage="hidden verification",
    )

    command_id = canonical_hidden_specification.command_id
    execution_result = evaluator_command_gateway.execute(command_id)
    if execution_result.command_id != command_id:
        _fail("hidden command gateway returned an inconsistent command ID")
    if execution_result.argv != resolved_hidden_command.command.argv:
        _fail("hidden command gateway returned inconsistent command arguments")
    verify_targets_unchanged(
        workspace=workspace,
        before=before,
        error_factory=HiddenVerificationError,
        stage="hidden verification",
    )
    verify_hidden_file_unchanged(resolved_hidden_command)
    evidence = ReproductionEvidence.from_execution_result(execution_result)
    status = hidden_status_for(evidence)
    return HiddenVerificationResult(
        task_id=canonical_task.task_id,
        task_fingerprint=task_fingerprint,
        hidden_specification_fingerprint=hidden_specification_fingerprint,
        reproduction_expectation_fingerprint=expectation_fingerprint,
        original_reproduction_run_fingerprint=run_fingerprint,
        proposal_digest=canonical_proposal.proposal_digest,
        regression_baseline_fingerprint=baseline_fingerprint,
        application_result_fingerprint=application_fingerprint,
        post_patch_reproduction_fingerprint=post_patch_fingerprint,
        regression_verification_fingerprint=regression_fingerprint,
        application_status=canonical_application.status,
        post_patch_reproduction_status=canonical_post_patch.status,
        regression_verification_status=canonical_regression.status,
        command_id=command_id,
        evidence=evidence,
        status=status,
        verification_summary=hidden_summary_for(status),
    )
