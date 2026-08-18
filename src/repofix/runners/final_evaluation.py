"""Path-based orchestration for pure terminal final evaluation."""

from pathlib import Path

from repofix.agent import ReproductionAgentRunResult
from repofix.final_evaluation import FinalEvaluationResult, finalize_evaluation
from repofix.hidden import HiddenVerificationResult
from repofix.patching import PatchApplicationResult, ValidatedPatchProposal
from repofix.policy import PolicyVerificationResult
from repofix.regression import RegressionBaselineResult, RegressionVerificationResult
from repofix.reproduction import PostPatchReproductionResult
from repofix.tasks import load_evaluator_task_bundle


def run_final_evaluation_from_paths(
    *,
    task_path: Path,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    baseline_result: RegressionBaselineResult,
    application_result: PatchApplicationResult,
    post_patch_reproduction_result: PostPatchReproductionResult,
    regression_verification_result: RegressionVerificationResult,
    hidden_verification_result: HiddenVerificationResult,
    policy_verification_result: PolicyVerificationResult,
) -> FinalEvaluationResult:
    """Load evaluator specifications once and finalize one successful artifact chain."""
    bundle = load_evaluator_task_bundle(task_path)
    task = bundle.agent_view()
    return finalize_evaluation(
        task=task,
        reproduction_expectation=bundle.reproduction,
        regression_specification=bundle.regression,
        hidden_specification=bundle.hidden_verification,
        policy_specification=bundle.patch_policy,
        original_reproduction_result=original_reproduction_result,
        proposal=proposal,
        regression_baseline_result=baseline_result,
        application_result=application_result,
        post_patch_reproduction_result=post_patch_reproduction_result,
        regression_verification_result=regression_verification_result,
        hidden_verification_result=hidden_verification_result,
        policy_verification_result=policy_verification_result,
    )
