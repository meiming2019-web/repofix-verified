"""Sanitized model context for patch proposal generation."""

from pydantic import Field

from repofix.agent import AgentPhase, AgentWorkflow, IssueUnderstanding, RepairHypothesis
from repofix.agent.reproduction_loop import ReproductionAgentRunResult, compute_task_fingerprint
from repofix.agent.state import REPRODUCED_TERMINAL_SUMMARY
from repofix.reproduction import ReproductionStatus
from repofix.tasks import AgentTaskSpec
from repofix.tasks.spec import StrictFrozenModel


MAX_PATCH_CONTEXT_FILE_CHARS = 8_000
MAX_PATCH_CONTEXT_TOTAL_FILE_CHARS = 20_000


class PatchContextFileObservation(StrictFrozenModel):
    path: str
    excerpt: str
    truncated: bool


class PatchProposalContext(StrictFrozenModel):
    task_id: str
    issue_title: str
    issue_body: str
    patchable_source_paths: tuple[str, ...]
    issue_understanding: IssueUnderstanding
    supported_hypotheses: tuple[RepairHypothesis, ...]
    successful_file_observations: tuple[PatchContextFileObservation, ...]
    reproduction_status: str = Field(pattern="^reproduced$")


class PatchProposalContextError(RuntimeError):
    """Raised when safe patch context cannot be constructed."""


def build_patch_proposal_context(
    *, task: AgentTaskSpec, reproduction_result: ReproductionAgentRunResult
) -> PatchProposalContext:
    state = reproduction_result.state
    if task.task_id != state.task_id:
        raise PatchProposalContextError("task and reproduction result IDs must match")
    if reproduction_result.task_fingerprint != compute_task_fingerprint(task):
        raise PatchProposalContextError("task fingerprint does not match reproduction result")
    if not task.patchable_source_paths:
        raise PatchProposalContextError("patch proposals require configured patchable source paths")
    if (
        state.workflow is not AgentWorkflow.REPRODUCTION
        or state.phase is not AgentPhase.FINISHED
        or state.terminal_summary != REPRODUCED_TERMINAL_SUMMARY
        or len(state.reproduction_observations) != 1
        or state.reproduction_observations[0].status is not ReproductionStatus.REPRODUCED
        or len(reproduction_result.attempts) != 1
        or reproduction_result.attempts[0].verdict.status is not ReproductionStatus.REPRODUCED
    ):
        raise PatchProposalContextError("patch proposals require completed verified reproduction")
    if state.issue_understanding is None:
        raise PatchProposalContextError("patch proposals require an issue understanding")
    hypotheses = tuple(h for h in state.hypotheses if h.status == "supported")
    hypothesis_ids = tuple(hypothesis.hypothesis_id for hypothesis in hypotheses)
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        raise PatchProposalContextError("patch context contains duplicate supported hypothesis IDs")
    if not hypotheses:
        raise PatchProposalContextError("patch proposals require a supported hypothesis")
    remaining = MAX_PATCH_CONTEXT_TOTAL_FILE_CHARS
    observations: list[PatchContextFileObservation] = []
    for observation in state.observations:
        if not observation.success or observation.tool_name != "read_file":
            continue
        path = observation.arguments.get("path")
        if not isinstance(path, str):
            continue
        limit = min(MAX_PATCH_CONTEXT_FILE_CHARS, remaining)
        excerpt = observation.output[:limit]
        observations.append(
            PatchContextFileObservation(
                path=path, excerpt=excerpt, truncated=len(excerpt) < len(observation.output)
            )
        )
        remaining -= len(excerpt)
    if not observations:
        raise PatchProposalContextError(
            "patch proposals require a successful read-file observation"
        )
    return PatchProposalContext(
        task_id=task.task_id,
        issue_title=task.issue_title,
        issue_body=task.issue_body,
        patchable_source_paths=task.patchable_source_paths,
        issue_understanding=state.issue_understanding,
        supported_hypotheses=hypotheses,
        successful_file_observations=tuple(observations),
        reproduction_status="reproduced",
    )
