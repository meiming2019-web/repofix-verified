"""Tests for strict agent state and action models."""

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from repofix.agent import (
    AgentAction,
    AgentPhase,
    AgentState,
    IssueUnderstanding,
    ReadFileAction,
    RepairHypothesis,
    ToolObservation,
)


ACTION_ADAPTER = TypeAdapter(AgentAction)


def understanding_data() -> dict[str, Any]:
    return {
        "expected_behavior": "The parser returns the configured value.",
        "observed_behavior": "The parser returns a default value.",
        "reproduction_clues": ["Failure occurs for an empty header."],
        "likely_components": ["src/repofix/parser.py"],
        "missing_information": [],
    }


def hypothesis_data() -> dict[str, Any]:
    return {
        "hypothesis_id": "hypothesis-1",
        "description": "An empty-header branch discards the configured value.",
        "supporting_evidence": ["The branch returns the default directly."],
        "contradicting_evidence": [],
        "confidence": 0.8,
        "status": "supported",
    }


def test_initial_state_construction() -> None:
    state = AgentState.initial("task-001")

    assert state.task_id == "task-001"
    assert state.phase is AgentPhase.UNDERSTAND
    assert state.issue_understanding is None
    assert state.hypotheses == ()
    assert state.observations == ()
    assert state.step_count == 0
    assert state.terminal_summary is None
    assert state.failure_reason is None


def state_data(
    *,
    phase: AgentPhase,
    terminal_summary: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, object]:
    return {
        "task_id": "task-001",
        "phase": phase,
        "issue_understanding": None,
        "hypotheses": (),
        "observations": (),
        "step_count": 0,
        "terminal_summary": terminal_summary,
        "failure_reason": failure_reason,
    }


def test_valid_finished_and_failed_states_are_accepted() -> None:
    finished = AgentState.model_validate(
        state_data(phase=AgentPhase.FINISHED, terminal_summary="Investigation complete.")
    )
    failed = AgentState.model_validate(
        state_data(phase=AgentPhase.FAILED, failure_reason="Step budget exhausted.")
    )

    assert finished.phase is AgentPhase.FINISHED
    assert finished.terminal_summary == "Investigation complete."
    assert failed.phase is AgentPhase.FAILED
    assert failed.failure_reason == "Step budget exhausted."


def test_finished_state_rejects_missing_terminal_summary() -> None:
    with pytest.raises(ValidationError):
        AgentState.model_validate(state_data(phase=AgentPhase.FINISHED))


def test_finished_state_rejects_failure_reason() -> None:
    with pytest.raises(ValidationError):
        AgentState.model_validate(
            state_data(
                phase=AgentPhase.FINISHED,
                terminal_summary="Investigation complete.",
                failure_reason="Contradictory failure.",
            )
        )


def test_failed_state_rejects_missing_failure_reason() -> None:
    with pytest.raises(ValidationError):
        AgentState.model_validate(state_data(phase=AgentPhase.FAILED))


def test_failed_state_rejects_terminal_summary() -> None:
    with pytest.raises(ValidationError):
        AgentState.model_validate(
            state_data(
                phase=AgentPhase.FAILED,
                terminal_summary="Contradictory summary.",
                failure_reason="Step budget exhausted.",
            )
        )


@pytest.mark.parametrize(
    "phase", [AgentPhase.UNDERSTAND, AgentPhase.EXPLORE, AgentPhase.HYPOTHESIZE]
)
def test_nonterminal_states_reject_terminal_summary(phase: AgentPhase) -> None:
    with pytest.raises(ValidationError):
        AgentState.model_validate(
            state_data(phase=phase, terminal_summary="Investigation complete.")
        )


@pytest.mark.parametrize(
    "phase", [AgentPhase.UNDERSTAND, AgentPhase.EXPLORE, AgentPhase.HYPOTHESIZE]
)
def test_nonterminal_states_reject_failure_reason(phase: AgentPhase) -> None:
    with pytest.raises(ValidationError):
        AgentState.model_validate(state_data(phase=phase, failure_reason="Unexpected failure."))


@pytest.mark.parametrize(
    ("phase", "terminal_summary", "failure_reason"),
    [
        (AgentPhase.FINISHED, "   ", None),
        (AgentPhase.FAILED, None, "\t"),
    ],
)
def test_terminal_states_reject_whitespace_only_result_text(
    phase: AgentPhase, terminal_summary: str | None, failure_reason: str | None
) -> None:
    with pytest.raises(ValidationError):
        AgentState.model_validate(
            state_data(
                phase=phase,
                terminal_summary=terminal_summary,
                failure_reason=failure_reason,
            )
        )


def test_issue_understanding_accepts_lists_and_stores_tuples() -> None:
    understanding = IssueUnderstanding.model_validate(understanding_data())

    assert understanding.reproduction_clues == ("Failure occurs for an empty header.",)
    assert understanding.likely_components == ("src/repofix/parser.py",)
    assert understanding.missing_information == ()


@pytest.mark.parametrize(
    "action_data",
    [
        {"kind": "understand_issue", "understanding": understanding_data()},
        {"kind": "list_files", "path": "src"},
        {"kind": "search_code", "query": "parse_header"},
        {
            "kind": "read_file",
            "path": "src/parser.py",
            "start_line": 1,
            "end_line": 20,
        },
        {"kind": "record_hypothesis", "hypothesis": hypothesis_data()},
        {"kind": "finish_investigation", "summary": "The likely branch was identified."},
    ],
)
def test_valid_discriminated_action_parsing(action_data: dict[str, Any]) -> None:
    action = ACTION_ADAPTER.validate_python(action_data)

    assert action.kind == action_data["kind"]


def test_rejects_unknown_action_kind() -> None:
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python({"kind": "execute_command", "command": "pytest"})


@pytest.mark.parametrize("line", [True, 1.5, "1", 0, -1])
def test_read_file_rejects_invalid_strict_line_numbers(line: object) -> None:
    with pytest.raises(ValidationError):
        ReadFileAction.model_validate(
            {"kind": "read_file", "path": "src/parser.py", "start_line": line, "end_line": 10}
        )


def test_read_file_rejects_reversed_line_range() -> None:
    with pytest.raises(ValidationError):
        ReadFileAction(
            kind="read_file", path="src/parser.py", start_line=20, end_line=10
        )


@pytest.mark.parametrize("confidence", [True, "0.8", -0.1, 1.1])
def test_repair_hypothesis_rejects_invalid_confidence(confidence: object) -> None:
    data = hypothesis_data()
    data["confidence"] = confidence

    with pytest.raises(ValidationError):
        RepairHypothesis.model_validate(data)


def test_tool_observation_enforces_success_error_consistency() -> None:
    with pytest.raises(ValidationError):
        ToolObservation(
            step_index=0,
            tool_name="search_code",
            arguments={},
            success=True,
            output="match",
            error="unexpected error",
        )

    with pytest.raises(ValidationError):
        ToolObservation(
            step_index=0,
            tool_name="search_code",
            arguments={},
            success=False,
            output="",
            error=" ",
        )
