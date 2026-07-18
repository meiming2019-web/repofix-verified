"""Path-based orchestration for post-patch reproduction verification."""

from pathlib import Path

from repofix.agent.reproduction_loop import (
    ReproductionAgentRunResult,
    compute_task_fingerprint,
)
from repofix.execution import LocalApprovedCommandExecutor
from repofix.patching import PatchApplicationResult, ValidatedPatchProposal
from repofix.reproduction.post_patch import (
    PostPatchReproductionResult,
    verify_post_patch_reproduction,
)
from repofix.reproduction.models import compute_reproduction_expectation_fingerprint
from repofix.tasks import load_reproduction_task_bundle


def run_post_patch_reproduction_from_paths(
    *,
    task_path: Path,
    workspace_root: Path,
    original_reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
    application_result: PatchApplicationResult,
) -> PostPatchReproductionResult:
    """Load the current bundle and rerun its reproduction command exactly once."""
    bundle = load_reproduction_task_bundle(task_path)
    task = bundle.agent_view()
    if original_reproduction_result.state.task_id != task.task_id:
        raise ValueError("original reproduction result does not belong to the current task")
    if original_reproduction_result.task_fingerprint != compute_task_fingerprint(task):
        raise ValueError("original reproduction result task fingerprint does not match")
    expectation_fingerprint = compute_reproduction_expectation_fingerprint(bundle.reproduction)
    if (
        original_reproduction_result.reproduction_expectation_fingerprint
        != expectation_fingerprint
    ):
        raise ValueError("original reproduction result expectation fingerprint does not match")
    command_gateway = LocalApprovedCommandExecutor(
        workspace_root=workspace_root,
        approved_commands=task.approved_commands,
        timeout_seconds=task.timeout_seconds,
    )
    return verify_post_patch_reproduction(
        workspace_root=workspace_root,
        task=task,
        expectation=bundle.reproduction,
        original_reproduction_result=original_reproduction_result,
        proposal=proposal,
        application_result=application_result,
        command_gateway=command_gateway,
    )
