"""Pure strict-success-only final evaluation."""

from __future__ import annotations

from typing import Never, TypeVar

from pydantic import BaseModel, ValidationError

from repofix.agent.reproduction_loop import (
    ReproductionAgentRunResult,
    compute_reproduction_run_fingerprint,
    compute_task_fingerprint,
)
from repofix.agent.state import AgentPhase
from repofix.final_evaluation.errors import FinalEvaluationError
from repofix.final_evaluation.models import (
    FINAL_EVALUATION_PASSED_SUMMARY,
    FinalEvaluationResult,
    FinalEvaluationStatus,
)
from repofix.hidden.models import (
    HiddenVerificationResult,
    HiddenVerificationStatus,
    compute_hidden_specification_fingerprint,
    compute_hidden_verification_fingerprint,
)
from repofix.patching.models import (
    PatchApplicationResult,
    PatchApplicationStatus,
    ValidatedPatchProposal,
    compute_patch_application_result_fingerprint,
)
from repofix.policy.models import (
    PolicyVerificationResult,
    PolicyVerificationStatus,
    compute_patch_policy_specification_fingerprint,
    compute_policy_verification_fingerprint,
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
    PatchPolicySpecification,
    RegressionSpecification,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _fail(message: str, cause: BaseException | None = None) -> Never:
    if cause is None:
        raise FinalEvaluationError(message)
    raise FinalEvaluationError(message) from cause


def _canonical(value: object, model: type[ModelT], description: str) -> ModelT:
    if not isinstance(value, model):
        _fail(f"{description} has an invalid type")
    try:
        return model.model_validate(value.model_dump())
    except ValidationError as error:
        _fail(f"{description} failed canonical integrity checks", error)


def _verify_application_metadata(
    *, proposal: ValidatedPatchProposal, application: PatchApplicationResult
) -> None:
    proposal_files = {item.path: item for item in proposal.file_snapshots}
    applied_files = {item.path: item for item in application.files}
    if set(proposal_files) != set(applied_files):
        _fail("final evaluation application paths do not match the proposal")
    for path in sorted(proposal_files):
        expected = proposal_files[path]
        actual = applied_files[path]
        if (
            actual.original_file_sha256 != expected.original_file_sha256
            or actual.original_size_bytes != expected.size_bytes
            or actual.candidate_file_sha256 != expected.candidate_file_sha256
            or actual.candidate_size_bytes != expected.candidate_size_bytes
        ):
            _fail("final evaluation application metadata does not match the proposal")


def _verify_chain(
    *,
    task: AgentTaskSpec,
    expectation: ReproductionExpectation,
    regression_specification: RegressionSpecification,
    hidden_specification: HiddenVerificationSpecification,
    policy_specification: PatchPolicySpecification,
    original: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    baseline: RegressionBaselineResult,
    application: PatchApplicationResult,
    post_patch: PostPatchReproductionResult,
    regression: RegressionVerificationResult,
    hidden: HiddenVerificationResult,
    policy: PolicyVerificationResult,
) -> tuple[str, str]:
    task_fingerprint = compute_task_fingerprint(task)
    expectation_fingerprint = compute_reproduction_expectation_fingerprint(expectation)
    original_fingerprint = compute_reproduction_run_fingerprint(original)
    regression_specification_fingerprint = (
        compute_regression_specification_fingerprint(regression_specification)
    )
    hidden_specification_fingerprint = compute_hidden_specification_fingerprint(
        hidden_specification
    )
    policy_specification_fingerprint = (
        compute_patch_policy_specification_fingerprint(policy_specification)
    )
    baseline_fingerprint = compute_regression_baseline_fingerprint(baseline)
    application_fingerprint = compute_patch_application_result_fingerprint(application)
    post_patch_fingerprint = compute_post_patch_reproduction_fingerprint(post_patch)
    regression_fingerprint = compute_regression_verification_fingerprint(regression)
    hidden_fingerprint = compute_hidden_verification_fingerprint(hidden)
    policy_fingerprint = compute_policy_verification_fingerprint(policy)

    if hidden_specification.command_id in task.approved_commands:
        _fail("final evaluation hidden command must not be agent-visible")
    if expectation.command_id not in task.approved_commands:
        _fail("final evaluation reproduction command is not approved")
    if regression_specification.command_id not in task.approved_commands:
        _fail("final evaluation regression command is not approved")
    if any(
        value != task.task_id
        for value in (
            original.state.task_id,
            proposal.task_id,
            baseline.task_id,
            application.task_id,
            post_patch.task_id,
            regression.task_id,
            hidden.task_id,
            policy.task_id,
        )
    ):
        _fail("final evaluation task identity does not match")
    if any(
        value != task_fingerprint
        for value in (
            original.task_fingerprint,
            proposal.task_fingerprint,
            baseline.task_fingerprint,
            application.task_fingerprint,
            post_patch.task_fingerprint,
            regression.task_fingerprint,
            hidden.task_fingerprint,
            policy.task_fingerprint,
        )
    ):
        _fail("final evaluation task fingerprint does not match")
    if any(
        value != expectation_fingerprint
        for value in (
            original.reproduction_expectation_fingerprint,
            proposal.reproduction_expectation_fingerprint,
            baseline.reproduction_expectation_fingerprint,
            application.reproduction_expectation_fingerprint,
            post_patch.reproduction_expectation_fingerprint,
            regression.reproduction_expectation_fingerprint,
            hidden.reproduction_expectation_fingerprint,
        )
    ):
        _fail("final evaluation reproduction expectation fingerprint does not match")
    if any(
        value != original_fingerprint
        for value in (
            proposal.reproduction_run_fingerprint,
            baseline.original_reproduction_run_fingerprint,
            application.reproduction_run_fingerprint,
            post_patch.original_reproduction_run_fingerprint,
            regression.original_reproduction_run_fingerprint,
            hidden.original_reproduction_run_fingerprint,
        )
    ):
        _fail("final evaluation original reproduction fingerprint does not match")
    if any(
        value != proposal.proposal_digest
        for value in (
            baseline.proposal_digest,
            application.proposal_digest,
            post_patch.proposal_digest,
            regression.proposal_digest,
            hidden.proposal_digest,
            policy.proposal_digest,
        )
    ):
        _fail("final evaluation proposal digest does not match")
    if (
        baseline.regression_specification_fingerprint
        != regression_specification_fingerprint
        or regression.regression_specification_fingerprint
        != regression_specification_fingerprint
    ):
        _fail("final evaluation regression specification fingerprint does not match")
    if hidden.hidden_specification_fingerprint != hidden_specification_fingerprint:
        _fail("final evaluation hidden specification fingerprint does not match")
    if policy.policy_specification_fingerprint != policy_specification_fingerprint:
        _fail("final evaluation policy specification fingerprint does not match")
    _verify_application_metadata(proposal=proposal, application=application)
    if (
        regression.regression_baseline_fingerprint != baseline_fingerprint
        or hidden.regression_baseline_fingerprint != baseline_fingerprint
    ):
        _fail("final evaluation regression baseline fingerprint does not match")
    if any(
        value != application_fingerprint
        for value in (
            regression.application_result_fingerprint,
            hidden.application_result_fingerprint,
            policy.application_result_fingerprint,
        )
    ):
        _fail("final evaluation application fingerprint does not match")
    if any(
        value != post_patch_fingerprint
        for value in (
            regression.post_patch_reproduction_fingerprint,
            hidden.post_patch_reproduction_fingerprint,
            policy.post_patch_reproduction_fingerprint,
        )
    ):
        _fail("final evaluation post-patch fingerprint does not match")
    if (
        hidden.regression_verification_fingerprint != regression_fingerprint
        or policy.regression_verification_fingerprint != regression_fingerprint
    ):
        _fail("final evaluation regression-verification fingerprint does not match")
    if policy.hidden_verification_fingerprint != hidden_fingerprint:
        _fail("final evaluation hidden-verification fingerprint does not match")

    if original.state.reproduction_command_id != expectation.command_id:
        _fail("final evaluation original reproduction command does not match")
    if len(original.attempts) != 1:
        _fail("final evaluation requires exactly one original reproduction attempt")
    original_evidence = original.attempts[0].evidence
    if original_evidence.argv != task.approved_commands[expectation.command_id].argv:
        _fail("final evaluation original reproduction arguments do not match")
    if baseline.command_id != regression_specification.command_id:
        _fail("final evaluation baseline command does not match")
    if regression.command_id != regression_specification.command_id:
        _fail("final evaluation regression command does not match")
    regression_argv = task.approved_commands[regression_specification.command_id].argv
    if baseline.evidence.argv != regression_argv or regression.evidence.argv != regression_argv:
        _fail("final evaluation regression arguments do not match")
    if post_patch.command_id != expectation.command_id:
        _fail("final evaluation post-patch reproduction command does not match")
    if post_patch.evidence.argv != task.approved_commands[expectation.command_id].argv:
        _fail("final evaluation post-patch reproduction arguments do not match")
    if hidden.command_id != hidden_specification.command_id:
        _fail("final evaluation hidden command does not match")
    if (
        len(hidden.evidence.argv) != len(hidden_specification.launcher.argv) + 1
        or hidden.evidence.argv[:-1] != hidden_specification.launcher.argv
    ):
        _fail("final evaluation hidden command arguments do not match")

    if (
        original.state.phase is not AgentPhase.FINISHED
        or original.attempts[0].verdict.status is not ReproductionStatus.REPRODUCED
    ):
        _fail("final evaluation requires confirmed original reproduction")
    if baseline.status is not RegressionBaselineStatus.PASSED:
        _fail("final evaluation requires a passing regression baseline")
    if application.status is not PatchApplicationStatus.APPLIED:
        _fail("final evaluation requires an applied patch result")
    if any(
        value is not application.status
        for value in (
            post_patch.application_status,
            regression.application_status,
            hidden.application_status,
            policy.application_status,
        )
    ):
        _fail("final evaluation application statuses do not match")
    if (
        post_patch.status
        is not PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
    ):
        _fail("final evaluation requires the post-patch reproduction gate")
    if any(
        value is not post_patch.status
        for value in (
            regression.post_patch_reproduction_status,
            hidden.post_patch_reproduction_status,
            policy.post_patch_reproduction_status,
        )
    ):
        _fail("final evaluation post-patch statuses do not match")
    if regression.status is not RegressionVerificationStatus.REGRESSION_COMMAND_PASSED:
        _fail("final evaluation requires the public regression pass gate")
    if any(
        value is not regression.status
        for value in (
            hidden.regression_verification_status,
            policy.regression_verification_status,
        )
    ):
        _fail("final evaluation regression statuses do not match")
    if hidden.status is not HiddenVerificationStatus.HIDDEN_COMMAND_PASSED:
        _fail("final evaluation requires the hidden pass gate")
    if policy.hidden_verification_status is not hidden.status:
        _fail("final evaluation hidden statuses do not match")
    if policy.status is not PolicyVerificationStatus.POLICY_PASSED:
        _fail("final evaluation requires the policy pass gate")
    if policy.findings:
        _fail("final evaluation requires an empty policy finding set")
    return task_fingerprint, policy_fingerprint


def finalize_evaluation(
    *,
    task: AgentTaskSpec,
    reproduction_expectation: ReproductionExpectation,
    regression_specification: RegressionSpecification,
    hidden_specification: HiddenVerificationSpecification,
    policy_specification: PatchPolicySpecification,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    regression_baseline_result: RegressionBaselineResult,
    application_result: PatchApplicationResult,
    post_patch_reproduction_result: PostPatchReproductionResult,
    regression_verification_result: RegressionVerificationResult,
    hidden_verification_result: HiddenVerificationResult,
    policy_verification_result: PolicyVerificationResult,
) -> FinalEvaluationResult:
    """Issue one terminal certificate for a complete successful evaluator chain."""
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
    canonical_policy_specification = _canonical(
        policy_specification,
        PatchPolicySpecification,
        "patch policy specification",
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
    canonical_hidden = _canonical(
        hidden_verification_result,
        HiddenVerificationResult,
        "hidden verification result",
    )
    canonical_policy = _canonical(
        policy_verification_result,
        PolicyVerificationResult,
        "policy verification result",
    )
    task_fingerprint, policy_fingerprint = _verify_chain(
        task=canonical_task,
        expectation=canonical_expectation,
        regression_specification=canonical_regression_specification,
        hidden_specification=canonical_hidden_specification,
        policy_specification=canonical_policy_specification,
        original=canonical_original,
        proposal=canonical_proposal,
        baseline=canonical_baseline,
        application=canonical_application,
        post_patch=canonical_post_patch,
        regression=canonical_regression,
        hidden=canonical_hidden,
        policy=canonical_policy,
    )
    return FinalEvaluationResult(
        task_id=canonical_task.task_id,
        task_fingerprint=task_fingerprint,
        proposal_digest=canonical_proposal.proposal_digest,
        policy_verification_fingerprint=policy_fingerprint,
        status=FinalEvaluationStatus.EVALUATOR_PASSED,
        verification_summary=FINAL_EVALUATION_PASSED_SUMMARY,
    )
