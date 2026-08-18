"""Path-based orchestration for the pre-patch regression baseline."""

from pathlib import Path

from repofix.agent.reproduction_loop import ReproductionAgentRunResult
from repofix.execution import LocalApprovedCommandExecutor, LocalExecutionContext
from repofix.patching import ValidatedPatchProposal
from repofix.regression import RegressionBaselineResult, establish_regression_baseline
from repofix.tasks import load_evaluator_task_bundle


def run_regression_baseline_from_paths(
    *,
    task_path: Path,
    workspace_root: Path,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    execution_context: LocalExecutionContext | None = None,
) -> RegressionBaselineResult:
    """Load evaluator data and execute its regression command once before patching."""
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
    return establish_regression_baseline(
        workspace_root=workspace_root,
        task=task,
        reproduction_expectation=bundle.reproduction,
        regression_specification=bundle.regression,
        original_reproduction_result=original_reproduction_result,
        proposal=proposal,
        command_gateway=command_gateway,
    )
