"""Path-based orchestration for post-patch regression verification."""

from pathlib import Path

from repofix.agent.reproduction_loop import ReproductionAgentRunResult
from repofix.execution import LocalApprovedCommandExecutor, LocalExecutionContext
from repofix.patching import PatchApplicationResult, ValidatedPatchProposal
from repofix.regression import (
    RegressionBaselineResult,
    RegressionVerificationResult,
    verify_post_patch_regression,
)
from repofix.reproduction import PostPatchReproductionResult
from repofix.tasks import load_evaluator_task_bundle


def run_regression_verification_from_paths(
    *,
    task_path: Path,
    workspace_root: Path,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    baseline_result: RegressionBaselineResult,
    application_result: PatchApplicationResult,
    post_patch_reproduction_result: PostPatchReproductionResult,
    execution_context: LocalExecutionContext | None = None,
) -> RegressionVerificationResult:
    """Load evaluator data and execute its post-patch regression command once."""
    bundle = load_evaluator_task_bundle(task_path)
    task = bundle.agent_view()
    if execution_context is None:
        command_gateway = LocalApprovedCommandExecutor(
            workspace_root=workspace_root,
            approved_commands=task.approved_commands,
            timeout_seconds=task.timeout_seconds,
        )
    else:
        command_gateway = LocalApprovedCommandExecutor(
            workspace_root=workspace_root,
            approved_commands=task.approved_commands,
            timeout_seconds=task.timeout_seconds,
            execution_context=execution_context,
        )
    return verify_post_patch_regression(
        workspace_root=workspace_root,
        task=task,
        reproduction_expectation=bundle.reproduction,
        regression_specification=bundle.regression,
        original_reproduction_result=original_reproduction_result,
        proposal=proposal,
        baseline_result=baseline_result,
        application_result=application_result,
        post_patch_reproduction_result=post_patch_reproduction_result,
        command_gateway=command_gateway,
    )
