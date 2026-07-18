"""Explicit read-only investigation loop."""

from repofix.agent.actions import (
    AgentAction,
    FinishInvestigationAction,
    ListFilesAction,
    ReadFileAction,
    RecordHypothesisAction,
    SearchCodeAction,
    UnderstandIssueAction,
)
from repofix.agent.interfaces import AgentModel, ReadOnlyToolGateway, ToolExecutionError
from repofix.agent.state import AgentPhase, AgentState, ToolObservation
from repofix.tasks import AgentTaskSpec


ReadOnlyToolAction = ListFilesAction | SearchCodeAction | ReadFileAction


class AgentProtocolError(ValueError):
    """Raised when a model action violates the investigation protocol."""


def _validated_state_update(state: AgentState, **updates: object) -> AgentState:
    values: dict[str, object] = {
        "task_id": state.task_id,
        "phase": state.phase,
        "issue_understanding": state.issue_understanding,
        "hypotheses": state.hypotheses,
        "observations": state.observations,
        "step_count": state.step_count,
        "terminal_summary": state.terminal_summary,
        "failure_reason": state.failure_reason,
    }
    values.update(updates)
    return AgentState.model_validate(values)


def _execute_tool_action(
    *, action: ReadOnlyToolAction, tools: ReadOnlyToolGateway, step_index: int
) -> ToolObservation:
    if isinstance(action, ListFilesAction):
        tool_name = "list_files"
        arguments: dict[str, object] = {"path": action.path}
    elif isinstance(action, SearchCodeAction):
        tool_name = "search_code"
        arguments = {"query": action.query, "file_glob": action.file_glob}
    else:
        tool_name = "read_file"
        arguments = {
            "path": action.path,
            "start_line": action.start_line,
            "end_line": action.end_line,
        }

    try:
        if isinstance(action, ListFilesAction):
            output = tools.list_files(action.path)
        elif isinstance(action, SearchCodeAction):
            output = tools.search_code(action.query, action.file_glob)
        else:
            read_result = tools.read_file_with_metadata(
                action.path, action.start_line, action.end_line
            )
            output = read_result.output
    except ToolExecutionError as error:
        return ToolObservation(
            step_index=step_index,
            tool_name=tool_name,
            arguments=arguments,
            success=False,
            output="",
            error=f"{type(error).__name__}: {error}",
            full_file_sha256=None,
        )
    return ToolObservation(
        step_index=step_index,
        tool_name=tool_name,
        arguments=arguments,
        success=True,
        output=output,
        error=None,
        full_file_sha256=(
            read_result.full_file_sha256 if isinstance(action, ReadFileAction) else None
        ),
    )


def _invalid_action(action: AgentAction, phase: AgentPhase) -> AgentProtocolError:
    return AgentProtocolError(f"action {action.kind!r} is not permitted in phase {phase.value}")


def run_read_only_investigation(
    *,
    task: AgentTaskSpec,
    model: AgentModel,
    tools: ReadOnlyToolGateway,
    max_steps: int = 12,
) -> AgentState:
    """Run a bounded investigation that can only observe repository state."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a strict positive integer")

    state = AgentState.initial(task.task_id)
    for _ in range(max_steps):
        action = model.next_action(task=task, state=state)
        next_step_count = state.step_count + 1

        if state.phase is AgentPhase.UNDERSTAND:
            if not isinstance(action, UnderstandIssueAction):
                raise _invalid_action(action, state.phase)
            state = _validated_state_update(
                state,
                phase=AgentPhase.EXPLORE,
                issue_understanding=action.understanding,
                step_count=next_step_count,
            )
            continue

        if isinstance(action, (ListFilesAction, SearchCodeAction, ReadFileAction)):
            if state.phase not in {AgentPhase.EXPLORE, AgentPhase.HYPOTHESIZE}:
                raise _invalid_action(action, state.phase)
            observation = _execute_tool_action(
                action=action, tools=tools, step_index=state.step_count
            )
            state = _validated_state_update(
                state,
                phase=AgentPhase.EXPLORE,
                observations=(*state.observations, observation),
                step_count=next_step_count,
            )
            continue

        if isinstance(action, RecordHypothesisAction):
            if state.phase not in {AgentPhase.EXPLORE, AgentPhase.HYPOTHESIZE}:
                raise _invalid_action(action, state.phase)
            state = _validated_state_update(
                state,
                phase=AgentPhase.HYPOTHESIZE,
                hypotheses=(*state.hypotheses, action.hypothesis),
                step_count=next_step_count,
            )
            continue

        if isinstance(action, FinishInvestigationAction):
            if state.phase is not AgentPhase.HYPOTHESIZE:
                raise _invalid_action(action, state.phase)
            if state.issue_understanding is None:
                raise AgentProtocolError("cannot finish without an issue understanding")
            if not state.observations:
                raise AgentProtocolError("cannot finish without a tool observation")
            if not state.hypotheses:
                raise AgentProtocolError("cannot finish without a repair hypothesis")
            return _validated_state_update(
                state,
                phase=AgentPhase.FINISHED,
                step_count=next_step_count,
                terminal_summary=action.summary,
            )

        raise _invalid_action(action, state.phase)

    return _validated_state_update(
        state,
        phase=AgentPhase.FAILED,
        failure_reason=f"investigation exceeded the {max_steps}-step budget",
    )
