"""Tests for application-level read-only investigation orchestration."""

from pathlib import Path

import pytest

import repofix.models.openai_agent as openai_agent_module
import repofix.runners.investigation as investigation_module
from repofix.agent import (
    AgentAction,
    AgentPhase,
    AgentState,
    FinishInvestigationAction,
    IssueUnderstanding,
    ReadFileAction,
    RecordHypothesisAction,
    RepairHypothesis,
    SearchCodeAction,
    ToolObservation,
    UnderstandIssueAction,
)
from repofix.runners import render_investigation_report, run_investigation_from_paths
from repofix.tasks import AgentTaskSpec


def write_task(path: Path) -> None:
    path.write_text(
        """task_id: runner-task
repository_url: https://github.com/example/runner-task.git
pre_fix_commit: 0123456789abcdef0123456789abcdef01234567
issue_title: Empty headers return the default
issue_body: Empty headers should retain the configured value.
approved_commands:
  unit_tests:
    argv: [pytest, -q]
allowed_source_paths: [src]
timeout_seconds: 300
""",
        encoding="utf-8",
    )


class ScriptedRunnerModel:
    def __init__(self) -> None:
        self.call_count = 0
        self.saw_real_observation = False

    def next_action(self, *, task: AgentTaskSpec, state: AgentState) -> AgentAction:
        assert task.task_id == "runner-task"
        turn = self.call_count
        self.call_count += 1
        if turn == 0:
            return UnderstandIssueAction(
                kind="understand_issue",
                understanding=IssueUnderstanding(
                    expected_behavior="Empty headers retain the configured value.",
                    observed_behavior="Empty headers return the default.",
                    reproduction_clues=("The issue describes the empty-header case.",),
                    likely_components=("src/parser.py",),
                    missing_information=(),
                ),
            )
        if turn == 1:
            return SearchCodeAction(kind="search_code", query="parse_header", file_glob="*.py")
        if turn == 2:
            observation = state.observations[-1]
            assert "src/parser.py:1:def parse_header" in observation.output
            assert "private.py" not in observation.output
            self.saw_real_observation = True
            return ReadFileAction(
                kind="read_file", path="src/parser.py", start_line=1, end_line=3
            )
        if turn == 3:
            observation = state.observations[-1]
            assert "return DEFAULT" in observation.output
            return RecordHypothesisAction(
                kind="record_hypothesis",
                hypothesis=RepairHypothesis(
                    hypothesis_id="premature-default",
                    description="The empty-header branch returns DEFAULT too early.",
                    supporting_evidence=(observation.output,),
                    contradicting_evidence=(),
                    confidence=0.9,
                    status="supported",
                ),
            )
        assert turn == 4
        return FinishInvestigationAction(
            kind="finish_investigation",
            summary="The premature default-return branch is the likely cause.",
        )


def test_runner_loads_task_uses_real_gateway_and_preserves_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_path = tmp_path / "task.yaml"
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "private").mkdir()
    source = workspace / "src/parser.py"
    private = workspace / "private/private.py"
    source.write_text(
        "def parse_header(header):\n    if not header:\n        return DEFAULT\n",
        encoding="utf-8",
    )
    private.write_text("def parse_header_private():\n    pass\n", encoding="utf-8")
    write_task(task_path)
    before = {source: source.read_bytes(), private: private.read_bytes()}

    loaded_paths: list[Path] = []
    real_loader = investigation_module.load_agent_task_spec

    def recording_loader(path: Path) -> AgentTaskSpec:
        loaded_paths.append(path)
        return real_loader(path)

    def reject_openai_construction(*args: object, **kwargs: object) -> object:
        raise AssertionError("the provider-independent runner constructed OpenAI")

    monkeypatch.setattr(investigation_module, "load_agent_task_spec", recording_loader)
    monkeypatch.setattr(openai_agent_module, "OpenAI", reject_openai_construction)
    model = ScriptedRunnerModel()

    state = run_investigation_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        model=model,
        max_steps=5,
    )

    assert state.phase is AgentPhase.FINISHED
    assert loaded_paths == [task_path]
    assert model.call_count == 5
    assert model.saw_real_observation is True
    assert [observation.tool_name for observation in state.observations] == [
        "search_code",
        "read_file",
    ]
    assert "private" not in repr(state.model_dump())
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("max_steps", [True, 0, -1, 1.5, "8", 21])
def test_runner_rejects_invalid_manual_step_limits(
    tmp_path: Path, max_steps: object
) -> None:
    with pytest.raises(ValueError, match="strict integer from 1 through 20"):
        run_investigation_from_paths(
            task_path=tmp_path / "unused.yaml",
            workspace_root=tmp_path,
            model=ScriptedRunnerModel(),
            max_steps=max_steps,  # type: ignore[arg-type]
        )


def report_state() -> AgentState:
    return AgentState(
        task_id="report-task",
        phase=AgentPhase.FINISHED,
        issue_understanding=IssueUnderstanding(
            expected_behavior="Expected value is retained.",
            observed_behavior="Default value is returned.",
            reproduction_clues=("Empty input triggers the issue.",),
            likely_components=("src/parser.py",),
            missing_information=(),
        ),
        observations=(
            ToolObservation(
                step_index=1,
                tool_name="search_code",
                arguments={"b": 2, "a": 1},
                success=True,
                output="src/parser.py:2:return DEFAULT",
                error=None,
            ),
            ToolObservation(
                step_index=2,
                tool_name="read_file",
                arguments={"path": "src/missing.py"},
                success=False,
                output="",
                error="ToolExecutionError: requested path does not exist",
            ),
        ),
        hypotheses=(
            RepairHypothesis(
                hypothesis_id="early-return",
                description="The empty-input branch returns the default too early.",
                supporting_evidence=("Observation 1 identifies the return.",),
                contradicting_evidence=(),
                confidence=0.85,
                status="supported",
            ),
        ),
        step_count=5,
        terminal_summary="Investigation identified the likely branch.",
        failure_reason=None,
    )


def test_report_is_deterministic_and_contains_public_investigation_details() -> None:
    state = report_state()

    first = render_investigation_report(state)
    second = render_investigation_report(state)

    assert first == second
    assert "Task ID: report-task" in first
    assert "Phase: FINISHED" in first
    assert "Steps: 5" in first
    assert "Expected behavior: Expected value is retained." in first
    assert first.index("1. Tool: search_code") < first.index("2. Tool: read_file")
    assert 'Arguments: {"a":1,"b":2}' in first
    assert "src/parser.py:2:return DEFAULT" in first
    assert "ToolExecutionError: requested path does not exist" in first
    assert "ID: early-return" in first
    assert "Confidence: 0.85" in first
    assert "Status: supported" in first
    assert "Final summary: Investigation identified the likely branch." in first
    assert "provider_response_id" not in first
    assert "hidden_tests" not in first
    assert "gold_patch" not in first
    assert "evaluator" not in first


def test_report_renders_failed_state_reason() -> None:
    state = AgentState(
        task_id="failed-task",
        phase=AgentPhase.FAILED,
        issue_understanding=None,
        hypotheses=(),
        observations=(),
        step_count=8,
        terminal_summary=None,
        failure_reason="investigation exceeded the 8-step budget",
    )

    report = render_investigation_report(state)

    assert "Phase: FAILED" in report
    assert "Failure reason: investigation exceeded the 8-step budget" in report


def test_report_truncates_display_without_mutating_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = report_state().model_copy(
        update={
            "observations": (
                report_state().observations[0].model_copy(update={"output": "ab\x07cd"}),
            )
        }
    )
    before = state.model_dump()
    monkeypatch.setattr(investigation_module, "MAX_REPORT_OBSERVATION_CHARS", 8)

    report = render_investigation_report(state)

    assert "     | ab\\u0007\n     | ...[observation output truncated]" in report
    assert "cd" not in report
    assert state.observations[0].output == "ab\x07cd"
    assert state.model_dump() == before


def test_report_escapes_controls_and_prevents_field_spoofing() -> None:
    original = report_state()
    assert original.issue_understanding is not None
    understanding = original.issue_understanding.model_copy(
        update={
            "expected_behavior": "ordinary 雪\x1b\x07\r\t\b\x7f\u0085\nPhase: FAILED",
            "observed_behavior": "still readable",
        }
    )
    successful = original.observations[0].model_copy(
        update={
            "tool_name": "search\nPhase: FAILED",
            "arguments": {"path": "雪\u0080\t"},
            "output": (
                "ordinary 雪\nFinal summary: verified\nPhase: FINISHED\n"
                "Error: success\x1b\x07\r\t\b\x7f\u0085"
            ),
        }
    )
    failed = original.observations[1].model_copy(
        update={"error": "failure\nFinal summary: forged-error"}
    )
    hypothesis = original.hypotheses[0].model_copy(
        update={
            "hypothesis_id": "candidate\nPhase: FAILED",
            "description": "description\nFinal summary: forged-description",
        }
    )
    state = original.model_copy(
        update={
            "task_id": "task\nFinal summary: forged-task",
            "issue_understanding": understanding,
            "observations": (successful, failed),
            "hypotheses": (hypothesis,),
            "terminal_summary": "done\nPhase: FAILED",
        }
    )
    before = state.model_dump()

    report = render_investigation_report(state)

    assert "ordinary 雪" in report
    for escaped in (
        "\\u001b",
        "\\u0007",
        "\\u000d",
        "\\u0009",
        "\\u0008",
        "\\u007f",
        "\\u0080",
        "\\u0085",
    ):
        assert escaped in report
    for raw_control in ("\x1b", "\x07", "\r", "\t", "\b", "\x7f", "\u0080", "\u0085"):
        assert raw_control not in report

    assert "     | ordinary 雪" in report
    assert "\n     | Final summary: verified" in report
    assert "\n     | Phase: FINISHED" in report
    assert "\n     | Error: success" in report
    assert "\nFinal summary: verified" not in report
    assert "Task ID: task\\u000aFinal summary: forged-task" in report
    assert "Final summary: done\\u000aPhase: FAILED" in report
    assert "failure\\u000aFinal summary: forged-error" in report
    assert "\nPhase: FAILED" not in report
    assert render_investigation_report(state) == report
    assert state.model_dump() == before
