"""Path-based orchestration for controlled patch application."""

from pathlib import Path

from repofix.agent.reproduction_loop import (
    ReproductionAgentRunResult,
    compute_task_fingerprint,
)
from repofix.patching import (
    PatchApplicationResult,
    ValidatedPatchProposal,
    apply_validated_patch_proposal,
)
from repofix.reproduction import compute_reproduction_expectation_fingerprint
from repofix.tasks import load_reproduction_task_bundle


def run_patch_application_from_paths(
    *,
    task_path: Path,
    workspace_root: Path,
    reproduction_result: ReproductionAgentRunResult,
    proposal: ValidatedPatchProposal,
) -> PatchApplicationResult:
    """Load the trusted bundle and apply one already validated proposal."""
    bundle = load_reproduction_task_bundle(task_path)
    task = bundle.agent_view()
    if reproduction_result.state.task_id != task.task_id:
        raise ValueError("reproduction result does not belong to the patch application task")
    if reproduction_result.task_fingerprint != compute_task_fingerprint(task):
        raise ValueError("reproduction result task fingerprint does not match")
    expectation_fingerprint = compute_reproduction_expectation_fingerprint(bundle.reproduction)
    if reproduction_result.reproduction_expectation_fingerprint != expectation_fingerprint:
        raise ValueError("reproduction result expectation fingerprint does not match")
    return apply_validated_patch_proposal(
        workspace_root=workspace_root,
        task=task,
        reproduction_result=reproduction_result,
        proposal=proposal,
    )
