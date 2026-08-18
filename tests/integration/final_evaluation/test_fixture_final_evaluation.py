"""Complete strict-success-only final evaluation for the canonical fixture."""

from pathlib import Path

from repofix.agent import AgentPhase
from repofix.final_evaluation import (
    FINAL_EVALUATION_PASSED_SUMMARY,
    FinalEvaluationStatus,
)
from repofix.hidden import HiddenVerificationStatus
from repofix.patching import PatchApplicationStatus
from repofix.policy import (
    PolicyVerificationStatus,
    compute_policy_verification_fingerprint,
)
from repofix.regression import RegressionBaselineStatus, RegressionVerificationStatus
from repofix.reproduction import PostPatchReproductionStatus, ReproductionStatus
from repofix.runners import run_final_evaluation_from_paths
from tests.integration.hidden.test_fixture_hidden_verification import (
    _hidden,
    _public_regression,
)
from tests.integration.policy.test_fixture_policy_verification import _policy
from tests.integration.regression.test_fixture_regression_verification import _chain


def _workspace_contents(workspace: Path) -> dict[str, bytes]:
    return {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }


def test_complete_fixture_chain_produces_one_pure_evaluator_pass(
    tmp_path: Path,
) -> None:
    chain = _chain(
        tmp_path,
        start=9,
        end=9,
        replacement="        return configured_value\n",
    )
    regression = _public_regression(chain)
    hidden = _hidden(chain, regression)
    policy = _policy(chain, regression, hidden)
    workspace_before = _workspace_contents(chain.workspace)
    model_calls_before = (chain.reproduction_model.calls, chain.patch_model.calls)

    result = run_final_evaluation_from_paths(
        task_path=chain.task_path,
        original_reproduction_result=chain.original,  # type: ignore[arg-type]
        proposal=chain.proposal,  # type: ignore[arg-type]
        baseline_result=chain.baseline,  # type: ignore[arg-type]
        application_result=chain.application,  # type: ignore[arg-type]
        post_patch_reproduction_result=chain.post_patch,  # type: ignore[arg-type]
        regression_verification_result=regression,
        hidden_verification_result=hidden,
        policy_verification_result=policy,
    )

    assert chain.original.state.phase is AgentPhase.FINISHED  # type: ignore[union-attr]
    assert (
        chain.original.attempts[0].verdict.status  # type: ignore[union-attr]
        is ReproductionStatus.REPRODUCED
    )
    assert chain.baseline.status is RegressionBaselineStatus.PASSED  # type: ignore[union-attr]
    assert chain.application.status is PatchApplicationStatus.APPLIED  # type: ignore[union-attr]
    assert (
        chain.post_patch.status  # type: ignore[union-attr]
        is PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
    )
    assert regression.status is RegressionVerificationStatus.REGRESSION_COMMAND_PASSED
    assert hidden.status is HiddenVerificationStatus.HIDDEN_COMMAND_PASSED
    assert policy.status is PolicyVerificationStatus.POLICY_PASSED
    assert policy.findings == ()
    assert result.status is FinalEvaluationStatus.EVALUATOR_PASSED
    assert result.verification_summary == FINAL_EVALUATION_PASSED_SUMMARY
    assert result.task_fingerprint == chain.original.task_fingerprint  # type: ignore[union-attr]
    assert result.proposal_digest == chain.proposal.proposal_digest  # type: ignore[union-attr]
    assert result.policy_verification_fingerprint == (
        compute_policy_verification_fingerprint(policy)
    )
    assert _workspace_contents(chain.workspace) == workspace_before
    assert (chain.reproduction_model.calls, chain.patch_model.calls) == (
        model_calls_before
    ) == (4, 1)
