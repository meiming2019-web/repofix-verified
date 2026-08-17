"""System-owned pre-patch regression baseline establishment."""

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
from repofix.patching.models import ValidatedPatchProposal
from repofix.regression._workspace import (
    capture_original_targets,
    resolve_workspace,
    verify_targets_unchanged,
)
from repofix.regression.models import (
    RegressionBaselineResult,
    baseline_status_for,
    baseline_summary_for,
    compute_regression_specification_fingerprint,
)
from repofix.reproduction.models import (
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionStatus,
    compute_reproduction_expectation_fingerprint,
)
from repofix.tasks.spec import AgentTaskSpec, RegressionSpecification

if TYPE_CHECKING:
    from repofix.agent import ApprovedCommandGateway


class RegressionBaselineError(RuntimeError):
    """Raised when a valid pre-patch regression baseline cannot be established."""


def _fail(message: str, cause: BaseException | None = None) -> Never:
    if cause is None:
        raise RegressionBaselineError(message)
    raise RegressionBaselineError(message) from cause


def _canonical_inputs(
    *,
    task: AgentTaskSpec,
    expectation: ReproductionExpectation,
    specification: RegressionSpecification,
    original_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
) -> tuple[
    AgentTaskSpec,
    ReproductionExpectation,
    RegressionSpecification,
    ReproductionAgentRunResult,
    ValidatedPatchProposal,
]:
    try:
        return (
            AgentTaskSpec.model_validate(task.model_dump()),
            ReproductionExpectation.model_validate(expectation.model_dump()),
            RegressionSpecification.model_validate(specification.model_dump()),
            ReproductionAgentRunResult.model_validate(original_result.model_dump()),
            ValidatedPatchProposal.model_validate(proposal.model_dump()),
        )
    except (AttributeError, ValidationError) as error:
        _fail("regression baseline inputs failed canonical integrity checks", error)


def _verify_bindings(
    *,
    task: AgentTaskSpec,
    expectation: ReproductionExpectation,
    specification: RegressionSpecification,
    original_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
) -> tuple[str, str, str, str]:
    task_fingerprint = compute_task_fingerprint(task)
    expectation_fingerprint = compute_reproduction_expectation_fingerprint(expectation)
    run_fingerprint = compute_reproduction_run_fingerprint(original_result)
    specification_fingerprint = compute_regression_specification_fingerprint(specification)
    if original_result.state.task_id != task.task_id or proposal.task_id != task.task_id:
        _fail("regression baseline task identity does not match")
    if (
        original_result.task_fingerprint != task_fingerprint
        or proposal.task_fingerprint != task_fingerprint
    ):
        _fail("regression baseline task fingerprint does not match")
    if (
        original_result.reproduction_expectation_fingerprint != expectation_fingerprint
        or proposal.reproduction_expectation_fingerprint != expectation_fingerprint
    ):
        _fail("regression baseline expectation fingerprint does not match")
    if proposal.reproduction_run_fingerprint != run_fingerprint:
        _fail("regression baseline reproduction run fingerprint does not match")
    if (
        original_result.state.phase is not AgentPhase.FINISHED
        or len(original_result.attempts) != 1
        or original_result.attempts[0].verdict.status is not ReproductionStatus.REPRODUCED
    ):
        _fail("regression baseline requires completed original reproduction")
    if specification.command_id not in task.approved_commands:
        _fail("regression baseline command is not approved by the current task")
    return (
        task_fingerprint,
        expectation_fingerprint,
        run_fingerprint,
        specification_fingerprint,
    )


def establish_regression_baseline(
    *,
    workspace_root: Path,
    task: AgentTaskSpec,
    reproduction_expectation: ReproductionExpectation,
    regression_specification: RegressionSpecification,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    command_gateway: ApprovedCommandGateway,
) -> RegressionBaselineResult:
    """Execute the configured regression command once on the unapplied workspace."""
    (
        canonical_task,
        canonical_expectation,
        canonical_specification,
        canonical_original,
        canonical_proposal,
    ) = _canonical_inputs(
        task=task,
        expectation=reproduction_expectation,
        specification=regression_specification,
        original_result=original_reproduction_result,
        proposal=proposal,
    )
    (
        task_fingerprint,
        expectation_fingerprint,
        run_fingerprint,
        specification_fingerprint,
    ) = _verify_bindings(
        task=canonical_task,
        expectation=canonical_expectation,
        specification=canonical_specification,
        original_result=canonical_original,
        proposal=canonical_proposal,
    )
    workspace = resolve_workspace(
        workspace_root,
        error_type=RegressionBaselineError,
        stage="regression baseline",
    )
    before = capture_original_targets(
        workspace=workspace,
        proposal=canonical_proposal,
        error_factory=RegressionBaselineError,
        stage="regression baseline",
    )

    command_id = canonical_specification.command_id
    execution_result = command_gateway.execute(command_id)
    if execution_result.command_id != command_id:
        _fail("regression baseline command gateway returned an inconsistent command ID")
    if execution_result.argv != canonical_task.approved_commands[command_id].argv:
        _fail("regression baseline command gateway returned inconsistent arguments")
    verify_targets_unchanged(
        workspace=workspace,
        before=before,
        error_factory=RegressionBaselineError,
        stage="regression baseline",
    )
    evidence = ReproductionEvidence.from_execution_result(execution_result)
    status = baseline_status_for(evidence)
    return RegressionBaselineResult(
        task_id=canonical_task.task_id,
        task_fingerprint=task_fingerprint,
        regression_specification_fingerprint=specification_fingerprint,
        reproduction_expectation_fingerprint=expectation_fingerprint,
        original_reproduction_run_fingerprint=run_fingerprint,
        proposal_digest=canonical_proposal.proposal_digest,
        command_id=command_id,
        evidence=evidence,
        status=status,
        baseline_summary=baseline_summary_for(status),
    )
