"""Integration coverage for the checked-in read-only investigation fixture."""

from pathlib import Path
import subprocess

import pytest

import repofix.models.openai_agent as openai_agent_module
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
    UnderstandIssueAction,
)
from repofix.runners import run_investigation_from_paths
from repofix.tasks import AgentTaskSpec


class FixtureInvestigationModel:
    def __init__(self) -> None:
        self.call_count = 0
        self.search_output = ""
        self.source_output = ""
        self.test_output = ""

    def next_action(self, *, task: AgentTaskSpec, state: AgentState) -> AgentAction:
        assert task.task_id == "empty-header-bug"
        turn = self.call_count
        self.call_count += 1
        if turn == 0:
            return UnderstandIssueAction(
                kind="understand_issue",
                understanding=IssueUnderstanding(
                    expected_behavior="Empty headers retain the configured value.",
                    observed_behavior="Empty headers return the module default.",
                    reproduction_clues=("The checked-in test states the expected behavior.",),
                    likely_components=("src/header_parser.py", "tests/test_header_parser.py"),
                    missing_information=(),
                ),
            )
        if turn == 1:
            return SearchCodeAction(kind="search_code", query="parse_header", file_glob="*.py")
        if turn == 2:
            observation = state.observations[-1]
            assert observation.tool_name == "search_code"
            assert "src/header_parser.py" in observation.output
            assert "tests/test_header_parser.py" in observation.output
            self.search_output = observation.output
            return ReadFileAction(
                kind="read_file", path="src/header_parser.py", start_line=1, end_line=12
            )
        if turn == 3:
            observation = state.observations[-1]
            assert observation.tool_name == "read_file"
            assert "return DEFAULT_VALUE" in observation.output
            self.source_output = observation.output
            return ReadFileAction(
                kind="read_file",
                path="tests/test_header_parser.py",
                start_line=1,
                end_line=12,
            )
        if turn == 4:
            observation = state.observations[-1]
            assert observation.tool_name == "read_file"
            assert 'parse_header("", "configured") == "configured"' in observation.output
            self.test_output = observation.output
            return RecordHypothesisAction(
                kind="record_hypothesis",
                hypothesis=RepairHypothesis(
                    hypothesis_id="premature-default-return",
                    description=(
                        "The empty-header branch returns DEFAULT_VALUE before the configured "
                        "value can be retained."
                    ),
                    supporting_evidence=(
                        f"Source observation: {self.source_output}",
                        f"Test observation: {self.test_output}",
                    ),
                    contradicting_evidence=(),
                    confidence=0.95,
                    status="supported",
                ),
            )
        assert turn == 5
        return FinishInvestigationAction(
            kind="finish_investigation",
            summary="The source and test support the premature default-return hypothesis.",
        )


def test_checked_in_fixture_runs_through_real_read_only_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    task_path = repository_root / "examples/tasks/empty-header-bug.yaml"
    workspace = repository_root / "examples/fixtures/empty-header-bug"
    assert (workspace / "pytest.ini").read_text(encoding="utf-8") == (
        "[pytest]\npythonpath = src\ntestpaths = tests\n"
    )
    files = [workspace / "src/header_parser.py", workspace / "tests/test_header_parser.py"]
    before = {path: path.read_bytes() for path in files}

    def reject_openai_construction(*args: object, **kwargs: object) -> object:
        raise AssertionError("fixture integration test attempted an OpenAI call")

    def reject_command_execution(*args: object, **kwargs: object) -> object:
        raise AssertionError("fixture integration test attempted command execution")

    monkeypatch.setattr(openai_agent_module, "OpenAI", reject_openai_construction)
    monkeypatch.setattr(subprocess, "run", reject_command_execution)
    model = FixtureInvestigationModel()

    state = run_investigation_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        model=model,
        max_steps=6,
    )

    assert state.phase is AgentPhase.FINISHED
    assert state.step_count == 6
    assert model.call_count == 6
    assert "def parse_header" in model.search_output
    assert "return DEFAULT_VALUE" in model.source_output
    assert "test_empty_header_retains_configured_value" in model.test_output
    assert state.hypotheses[0].status == "supported"
    assert "Source observation:" in state.hypotheses[0].supporting_evidence[0]
    assert "Test observation:" in state.hypotheses[0].supporting_evidence[1]
    assert {path: path.read_bytes() for path in files} == before

    rendered_state = repr(state.model_dump())
    assert "pytest" not in rendered_state
    assert "approved_commands" not in rendered_state
    assert "patch" not in rendered_state
    assert "hidden_tests" not in rendered_state
    assert "gold_patch" not in rendered_state
