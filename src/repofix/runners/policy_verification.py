"""Path-based orchestration for protected workspace policy verification."""

from pathlib import Path

from repofix.hidden import HiddenVerificationResult
from repofix.patching import PatchApplicationResult, ValidatedPatchProposal
from repofix.policy import PolicyVerificationResult, verify_patch_policy
from repofix.regression import RegressionVerificationResult
from repofix.reproduction import PostPatchReproductionResult
from repofix.tasks import load_evaluator_task_bundle


def run_policy_verification_from_paths(
    *,
    task_path: Path,
    workspace_root: Path,
    proposal: ValidatedPatchProposal,
    application_result: PatchApplicationResult,
    post_patch_reproduction_result: PostPatchReproductionResult,
    regression_verification_result: RegressionVerificationResult,
    hidden_verification_result: HiddenVerificationResult,
) -> PolicyVerificationResult:
    """Load evaluator policy once and inspect the applied candidate without commands."""
    bundle = load_evaluator_task_bundle(task_path)
    task = bundle.agent_view()
    return verify_patch_policy(
        workspace_root=workspace_root,
        task=task,
        policy_specification=bundle.patch_policy,
        proposal=proposal,
        application_result=application_result,
        post_patch_reproduction_result=post_patch_reproduction_result,
        regression_verification_result=regression_verification_result,
        hidden_verification_result=hidden_verification_result,
    )
