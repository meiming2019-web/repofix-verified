"""Provider-independent orchestration for unapplied patch proposals."""

from pathlib import Path
from typing import Protocol

from repofix.agent.reproduction_loop import ReproductionAgentRunResult
from repofix.agent.reproduction_loop import compute_task_fingerprint
from repofix.patching import (
    PatchProposalContext,
    PatchProposalDraft,
    ValidatedPatchProposal,
    build_patch_proposal_context,
    validate_patch_proposal,
    validate_patch_workspace_reads,
)
from repofix.reproduction import compute_reproduction_expectation_fingerprint
from repofix.tasks import load_reproduction_task_bundle


class PatchProposalModel(Protocol):
    def propose_patch(self, *, context: PatchProposalContext) -> PatchProposalDraft: ...


def run_patch_proposal_from_paths(
    *,
    task_path: Path,
    workspace_root: Path,
    reproduction_result: ReproductionAgentRunResult,
    model: PatchProposalModel,
) -> ValidatedPatchProposal:
    bundle = load_reproduction_task_bundle(task_path)
    task = bundle.agent_view()
    if not task.patchable_source_paths:
        raise ValueError("patch proposal task must configure patchable source paths")
    if reproduction_result.state.task_id != task.task_id:
        raise ValueError("reproduction result does not belong to the patch proposal task")
    if reproduction_result.task_fingerprint != compute_task_fingerprint(task):
        raise ValueError("reproduction result task fingerprint does not match")
    expectation_fingerprint = compute_reproduction_expectation_fingerprint(bundle.reproduction)
    if reproduction_result.reproduction_expectation_fingerprint != expectation_fingerprint:
        raise ValueError("reproduction result expectation fingerprint does not match")
    validate_patch_workspace_reads(
        workspace_root=workspace_root,
        task=task,
        reproduction_result=reproduction_result,
    )
    context = build_patch_proposal_context(task=task, reproduction_result=reproduction_result)
    draft = model.propose_patch(context=context)
    return validate_patch_proposal(
        workspace_root=workspace_root,
        task=task,
        reproduction_result=reproduction_result,
        draft=draft,
    )
