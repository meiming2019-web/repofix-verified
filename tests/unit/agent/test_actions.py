"""Tests for strict agent state and action models."""

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from repofix.agent import (
    AgentAction,
    AgentPhase,
    AgentReproductionObservation,
    AgentState,
    AgentWorkflow,
    IssueUnderstanding,
    ReadFileAction,
    RepairHypothesis,
    RunApprovedCommandAction,
    ToolObservation,
)
from repofix.execution import CommandTerminationReason
from repofix.reproduction import ReproductionStatus
from repofix.agent.state import REPRODUCED_TERMINAL_SUMMARY


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
    assert state.workflow is AgentWorkflow.INVESTIGATION
    assert state.reproduction_observations == ()


def test_reproduction_initial_state_is_explicit() -> None:
    state = AgentState.initial(
        "task-001",
        workflow=AgentWorkflow.REPRODUCTION,
        reproduction_command_id="unit_tests",
    )

    assert state.workflow is AgentWorkflow.REPRODUCTION
    assert state.phase is AgentPhase.UNDERSTAND
    assert state.reproduction_command_id == "unit_tests"

    with pytest.raises(ValidationError, match="configured command ID"):
        AgentState.initial("task-001", workflow=AgentWorkflow.REPRODUCTION)
    with pytest.raises(ValidationError, match="investigation states"):
        AgentState.initial("task-001", reproduction_command_id="unit_tests")


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
        {"kind": "run_approved_command", "command_id": "unit_tests"},
        {"kind": "finish_investigation", "summary": "The likely branch was identified."},
    ],
)
def test_valid_discriminated_action_parsing(action_data: dict[str, Any]) -> None:
    action = ACTION_ADAPTER.validate_python(action_data)

    assert action.kind == action_data["kind"]


def test_rejects_unknown_action_kind() -> None:
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python({"kind": "execute_command", "command": "pytest"})


def test_run_approved_command_action_contains_only_exact_id() -> None:
    action = RunApprovedCommandAction(command_id="unit_tests")

    assert action.kind == "run_approved_command"
    assert action.model_dump() == {
        "kind": "run_approved_command",
        "command_id": "unit_tests",
    }


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "run_approved_command", "command_id": ""},
        {"kind": "run_approved_command", "command_id": "unit tests"},
        {"kind": "run_approved_command", "command_id": "unit_tests", "argv": ["pytest"]},
        {"kind": "run_approved_command", "command_id": "unit_tests", "shell": True},
    ],
)
def test_run_approved_command_rejects_invalid_or_extra_fields(
    data: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ACTION_ADAPTER.validate_python(data)


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

    with pytest.raises(ValidationError, match="full-file SHA-256"):
        ToolObservation(
            step_index=0,
            tool_name="read_file",
            arguments={"path": "src/a.py"},
            success=True,
            output="1: value\n",
            error=None,
        )

    with pytest.raises(ValidationError, match="only successful read-file"):
        ToolObservation(
            step_index=0,
            tool_name="search_code",
            arguments={"query": "value"},
            success=True,
            output="src/a.py:1:value",
            error=None,
            full_file_sha256="a" * 64,
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


def test_agent_state_rejects_duplicate_hypothesis_ids() -> None:
    supported = RepairHypothesis.model_validate(hypothesis_data())
    rejected = supported.model_copy(update={"description": "different cause", "status": "rejected"})
    data = state_data(phase=AgentPhase.HYPOTHESIZE)
    data["hypotheses"] = (supported, rejected)

    with pytest.raises(ValidationError, match="hypothesis IDs must be unique"):
        AgentState.model_validate(data)


def reproduction_observation(
    status: ReproductionStatus = ReproductionStatus.REPRODUCED,
) -> AgentReproductionObservation:
    return AgentReproductionObservation(
        command_id="unit_tests",
        termination_reason=CommandTerminationReason.COMPLETED,
        exit_code=1,
        stdout="bounded output",
        stderr="",
        stdout_bytes=14,
        stderr_bytes=0,
        had_decode_errors=False,
        status=status,
    )


@pytest.mark.parametrize(
    ("status", "reason", "exit_code"),
    [
        (ReproductionStatus.REPRODUCED, CommandTerminationReason.COMPLETED, 0),
        (ReproductionStatus.REPRODUCED, CommandTerminationReason.TIMED_OUT, None),
        (ReproductionStatus.NOT_REPRODUCED, CommandTerminationReason.COMPLETED, 1),
        (ReproductionStatus.NOT_REPRODUCED, CommandTerminationReason.OUTPUT_LIMIT, None),
    ],
)
def test_reproduction_observation_enforces_status_execution_invariants(
    status: ReproductionStatus,
    reason: CommandTerminationReason,
    exit_code: int | None,
) -> None:
    with pytest.raises(ValidationError):
        AgentReproductionObservation(
            command_id="unit_tests",
            termination_reason=reason,
            exit_code=exit_code,
            stdout="",
            stderr="",
            stdout_bytes=0,
            stderr_bytes=0,
            had_decode_errors=False,
            status=status,
        )


def test_sanitized_reproduction_observation_forbids_evaluator_fields() -> None:
    observation = reproduction_observation()

    assert set(observation.model_dump()) == {
        "command_id",
        "termination_reason",
        "exit_code",
        "stdout",
        "stderr",
        "stdout_bytes",
        "stderr_bytes",
        "had_decode_errors",
        "status",
    }
    for field in ("argv", "reasons", "matched_fragment_ids", "expected_exit_codes"):
        with pytest.raises(ValidationError):
            AgentReproductionObservation.model_validate(
                {**observation.model_dump(), field: "evaluator-only"}
            )


def test_obsolete_reproduced_phase_is_removed() -> None:
    assert "REPRODUCED" not in AgentPhase.__members__


def test_reproduced_observation_requires_finished_state() -> None:
    second = reproduction_observation(ReproductionStatus.REPRODUCED)
    base = {
        "task_id": "task-001",
        "issue_understanding": None,
        "hypotheses": (),
        "observations": (),
        "step_count": 1,
        "terminal_summary": None,
        "failure_reason": None,
        "workflow": AgentWorkflow.REPRODUCTION,
        "reproduction_command_id": "unit_tests",
        "reproduction_observations": (second,),
    }

    for phase in (AgentPhase.UNDERSTAND, AgentPhase.EXPLORE, AgentPhase.HYPOTHESIZE):
        with pytest.raises(ValidationError, match="requires a finished state"):
            AgentState.model_validate({**base, "phase": phase})
    with pytest.raises(ValidationError, match="requires a finished state"):
        AgentState.model_validate(
            {
                **base,
                "phase": AgentPhase.FAILED,
                "failure_reason": "Step budget exhausted.",
            }
        )


def test_reproduction_state_rejects_multiple_observations() -> None:
    first = reproduction_observation(ReproductionStatus.INCONCLUSIVE)
    second = reproduction_observation(ReproductionStatus.REPRODUCED)
    with pytest.raises(ValidationError, match="at most one"):
        AgentState(
            task_id="task-001",
            phase=AgentPhase.FAILED,
            issue_understanding=None,
            hypotheses=(),
            observations=(),
            step_count=2,
            terminal_summary=None,
            failure_reason="Step budget exhausted.",
            workflow=AgentWorkflow.REPRODUCTION,
            reproduction_command_id="unit_tests",
            reproduction_observations=(first, second),
        )
    with pytest.raises(ValidationError):
        AgentState(
            task_id="task-001",
            phase=AgentPhase.FAILED,
            issue_understanding=None,
            hypotheses=(),
            observations=(),
            step_count=1,
            terminal_summary=None,
            failure_reason="Step budget exhausted.",
            workflow=AgentWorkflow.INVESTIGATION,
            reproduction_command_id="unit_tests",
            reproduction_observations=(first,),
        )


@pytest.mark.parametrize(
    "summary",
    [
        "The behavior was not reproduced.",
        "The reported behavior was reproduced.",
        "The bug was fixed.",
    ],
)
def test_finished_reproduction_requires_canonical_summary(summary: str) -> None:
    with pytest.raises(ValidationError, match="canonical summary"):
        AgentState(
            task_id="task-001",
            phase=AgentPhase.FINISHED,
            issue_understanding=None,
            hypotheses=(),
            observations=(),
            step_count=1,
            terminal_summary=summary,
            failure_reason=None,
            workflow=AgentWorkflow.REPRODUCTION,
            reproduction_command_id="unit_tests",
            reproduction_observations=(reproduction_observation(),),
        )


def test_finished_reproduction_accepts_exact_canonical_summary() -> None:
    state = AgentState(
        task_id="task-001",
        phase=AgentPhase.FINISHED,
        issue_understanding=None,
        hypotheses=(),
        observations=(),
        step_count=1,
        terminal_summary=REPRODUCED_TERMINAL_SUMMARY,
        failure_reason=None,
        workflow=AgentWorkflow.REPRODUCTION,
        reproduction_command_id="unit_tests",
        reproduction_observations=(reproduction_observation(),),
    )

    assert state.terminal_summary == REPRODUCED_TERMINAL_SUMMARY
