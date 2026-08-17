"""Path-based orchestration for evaluator-only hidden verification."""

from pathlib import Path

from repofix.agent.reproduction_loop import ReproductionAgentRunResult
from repofix.execution import LocalApprovedCommandExecutor
from repofix.hidden import (
    HiddenVerificationResult,
    resolve_hidden_command,
    verify_hidden_behavior,
)
from repofix.patching import PatchApplicationResult, ValidatedPatchProposal
from repofix.regression import RegressionBaselineResult, RegressionVerificationResult
from repofix.reproduction import PostPatchReproductionResult
from repofix.tasks import load_evaluator_task_bundle


def run_hidden_verification_from_paths(
    *,
    task_path: Path,
    workspace_root: Path,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    baseline_result: RegressionBaselineResult,
    application_result: PatchApplicationResult,
    post_patch_reproduction_result: PostPatchReproductionResult,
    regression_verification_result: RegressionVerificationResult,
) -> HiddenVerificationResult:
    """Load evaluator data and execute its singleton hidden command once."""
    bundle = load_evaluator_task_bundle(task_path)
    task = bundle.agent_view()
    resolution = resolve_hidden_command(
        workspace_root=workspace_root,
        evaluator_assets_root=task_path.parent,
        specification=bundle.hidden_verification,
    )
    command_gateway = LocalApprovedCommandExecutor(
        workspace_root=workspace_root,
        approved_commands={resolution.command_id: resolution.command},
        timeout_seconds=task.timeout_seconds,
    )
    return verify_hidden_behavior(
        workspace_root=workspace_root,
        task=task,
        reproduction_expectation=bundle.reproduction,
        regression_specification=bundle.regression,
        hidden_specification=bundle.hidden_verification,
        original_reproduction_result=original_reproduction_result,
        proposal=proposal,
        regression_baseline_result=baseline_result,
        application_result=application_result,
        post_patch_reproduction_result=post_patch_reproduction_result,
        regression_verification_result=regression_verification_result,
        resolved_hidden_command=resolution,
        evaluator_command_gateway=command_gateway,
    )
