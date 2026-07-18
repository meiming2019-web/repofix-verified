"""Provider-independent orchestration for agent-requested reproduction."""

from pathlib import Path

from repofix.agent import AgentModel
from repofix.agent.reproduction_loop import (
    ReproductionAgentRunResult,
    run_reproduction_agent_loop,
)
from repofix.execution import LocalApprovedCommandExecutor
from repofix.tasks import load_reproduction_task_bundle
from repofix.tools import LocalReadOnlyToolGateway


MAX_REPRODUCTION_STEPS = 20


def run_reproduction_from_paths(
    *,
    task_path: Path,
    workspace_root: Path,
    model: AgentModel,
    max_steps: int,
) -> ReproductionAgentRunResult:
    """Load an evaluator bundle and run reproduction in a prepared workspace."""
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or not 1 <= max_steps <= MAX_REPRODUCTION_STEPS
    ):
        raise ValueError("max_steps must be a strict integer from 1 through 20")

    bundle = load_reproduction_task_bundle(task_path)
    task = bundle.agent_view()
    tools = LocalReadOnlyToolGateway(
        workspace_root=workspace_root,
        allowed_source_paths=task.allowed_source_paths,
    )
    command_gateway = LocalApprovedCommandExecutor(
        workspace_root=workspace_root,
        approved_commands=task.approved_commands,
        timeout_seconds=task.timeout_seconds,
    )
    return run_reproduction_agent_loop(
        task=task,
        expectation=bundle.reproduction,
        model=model,
        tools=tools,
        command_gateway=command_gateway,
        max_steps=max_steps,
    )
