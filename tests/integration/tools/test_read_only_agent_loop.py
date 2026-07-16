"""Integration test for the agent loop with real read-only repository tools."""

from pathlib import Path

import pytest

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
    ToolExecutionError,
    UnderstandIssueAction,
    run_read_only_investigation,
)
from repofix.tasks import AgentTaskSpec
from repofix.tools import LocalReadOnlyToolGateway


class RepositoryInvestigationModel:
    def __init__(self) -> None:
        self.call_count = 0
        self.saw_search_observation = False
        self.saw_read_observation = False

    def next_action(self, *, task: AgentTaskSpec, state: AgentState) -> AgentAction:
        assert task.task_id == "integration-task"
        turn = self.call_count
        self.call_count += 1
        if turn == 0:
            assert state.phase is AgentPhase.UNDERSTAND
            return UnderstandIssueAction(
                kind="understand_issue",
                understanding=IssueUnderstanding.model_validate(
                    {
                        "expected_behavior": "Empty headers retain the configured value.",
                        "observed_behavior": "Empty headers return a default value.",
                        "reproduction_clues": ["The parser test describes the failure."],
                        "likely_components": ["src/parser.py"],
                        "missing_information": [],
                    }
                ),
            )
        if turn == 1:
            assert state.phase is AgentPhase.EXPLORE
            return SearchCodeAction(kind="search_code", query="parse_header", file_glob="*.py")
        if turn == 2:
            observation = state.observations[-1]
            assert observation.tool_name == "search_code"
            assert "src/parser.py:1:def parse_header" in observation.output
            assert "private/secret.py" not in observation.output
            self.saw_search_observation = True
            return ReadFileAction(
                kind="read_file", path="src/parser.py", start_line=1, end_line=3
            )
        if turn == 3:
            observation = state.observations[-1]
            assert observation.tool_name == "read_file"
            assert "1: def parse_header(header):" in observation.output
            assert "2:     if not header:" in observation.output
            self.saw_read_observation = True
            return RecordHypothesisAction(
                kind="record_hypothesis",
                hypothesis=RepairHypothesis.model_validate(
                    {
                        "hypothesis_id": "empty-header-branch",
                        "description": "The empty-header branch returns DEFAULT too early.",
                        "supporting_evidence": [observation.output],
                        "contradicting_evidence": [],
                        "confidence": 0.9,
                        "status": "supported",
                    }
                ),
            )
        assert turn == 4
        assert state.phase is AgentPhase.HYPOTHESIZE
        assert state.hypotheses[-1].hypothesis_id == "empty-header-branch"
        return FinishInvestigationAction(
            kind="finish_investigation",
            summary="Read-only investigation identified the likely faulty branch.",
        )


def test_agent_loop_uses_real_read_only_repository_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "private").mkdir()
    (workspace / "src/real").mkdir()
    source = workspace / "src/parser.py"
    test_file = workspace / "tests/test_parser.py"
    private_file = workspace / "private/secret.py"
    linked_file = workspace / "src/real/linked.py"
    source.write_text(
        "def parse_header(header):\n    if not header:\n        return DEFAULT\n",
        encoding="utf-8",
    )
    test_file.write_text("def test_parse_header():\n    assert parse_header('')\n", encoding="utf-8")
    private_file.write_text("def parse_header_secret():\n    pass\n", encoding="utf-8")
    linked_file.write_text("linked content\n", encoding="utf-8")
    try:
        (workspace / "src/link").symlink_to(workspace / "src/real", target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symbolic links are not supported on this host: {error}")
    before = {
        path: path.read_bytes() for path in (source, test_file, private_file, linked_file)
    }

    task = AgentTaskSpec.model_validate(
        {
            "task_id": "integration-task",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0123456789abcdef0123456789abcdef01234567",
            "issue_title": "Empty headers return the wrong value",
            "issue_body": "The parser returns DEFAULT for an empty header.",
            "approved_commands": {"unit_tests": {"argv": ["pytest", "-q"]}},
            "allowed_source_paths": ["src", "tests"],
            "timeout_seconds": 300,
        }
    )
    model = RepositoryInvestigationModel()
    tools = LocalReadOnlyToolGateway(
        workspace_root=workspace,
        allowed_source_paths=task.allowed_source_paths,
    )
    with pytest.raises(ToolExecutionError, match="must not traverse"):
        tools.read_file("src/link/linked.py", 1, 1)

    state = run_read_only_investigation(task=task, model=model, tools=tools)

    assert state.phase is AgentPhase.FINISHED
    assert state.step_count == 5
    assert model.call_count == 5
    assert model.saw_search_observation is True
    assert model.saw_read_observation is True
    assert [observation.tool_name for observation in state.observations] == [
        "search_code",
        "read_file",
    ]
    assert state.observations[0].arguments == {"query": "parse_header", "file_glob": "*.py"}
    assert state.observations[1].arguments == {
        "path": "src/parser.py",
        "start_line": 1,
        "end_line": 3,
    }
    assert state.hypotheses[-1].hypothesis_id == "empty-header-branch"
    assert {path: path.read_bytes() for path in before} == before
    assert "private" not in repr(state.model_dump())
    assert "hidden_tests" not in repr(state.model_dump())
    assert "gold_patch" not in repr(state.model_dump())
