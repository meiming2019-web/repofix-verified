"""Integration coverage for the real fixture reproduction agent workflow."""

from pathlib import Path
import shutil

import pytest

import repofix.models.openai_agent as openai_agent_module
from repofix.agent import (
    AgentAction,
    AgentPhase,
    AgentState,
    AgentWorkflow,
    IssueUnderstanding,
    ReadFileAction,
    RecordHypothesisAction,
    RepairHypothesis,
    RunApprovedCommandAction,
    SearchCodeAction,
    UnderstandIssueAction,
)
from repofix.agent.prompts import build_investigation_messages
from repofix.reproduction import ReproductionStatus
from repofix.runners import run_reproduction_from_paths
from repofix.tasks import AgentTaskSpec


class FixtureReproductionModel:
    def __init__(self) -> None:
        self.call_count = 0
        self.prompt_renderings: list[str] = []

    def next_action(self, *, task: AgentTaskSpec, state: AgentState) -> AgentAction:
        self.prompt_renderings.append(repr(build_investigation_messages(task=task, state=state)))
        turn = self.call_count
        self.call_count += 1
        if turn == 0:
            return UnderstandIssueAction(
                kind="understand_issue",
                understanding=IssueUnderstanding(
                    expected_behavior="Empty headers retain the configured value.",
                    observed_behavior="Empty headers return the module default.",
                    reproduction_clues=("The issue identifies the empty-header case.",),
                    likely_components=("src/header_parser.py", "tests/test_header_parser.py"),
                    missing_information=(),
                ),
            )
        if turn == 1:
            return SearchCodeAction(
                kind="search_code", query="parse_header", file_glob="*.py"
            )
        if turn == 2:
            assert "src/header_parser.py" in state.observations[-1].output
            return ReadFileAction(
                kind="read_file", path="src/header_parser.py", start_line=1, end_line=12
            )
        if turn == 3:
            assert "return DEFAULT_VALUE" in state.observations[-1].output
            return ReadFileAction(
                kind="read_file",
                path="tests/test_header_parser.py",
                start_line=1,
                end_line=12,
            )
        if turn == 4:
            assert "test_empty_header_retains_configured_value" in state.observations[-1].output
            return RecordHypothesisAction(
                kind="record_hypothesis",
                hypothesis=RepairHypothesis(
                    hypothesis_id="premature-default-return",
                    description="The empty-header branch returns DEFAULT_VALUE too early.",
                    supporting_evidence=(
                        "The source and test observations identify the incorrect branch.",
                    ),
                    contradicting_evidence=(),
                    confidence=0.95,
                    status="supported",
                ),
            )
        if turn == 5:
            return RunApprovedCommandAction(command_id="unit_tests")
        raise AssertionError("model was called after verified reproduction")


def test_fixture_runs_through_agent_requested_reproduction_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    source_workspace = repository_root / "examples/fixtures/empty-header-bug"
    workspace = tmp_path / "empty-header-bug"
    shutil.copytree(
        source_workspace,
        workspace,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    source = workspace / "src/header_parser.py"
    test_file = workspace / "tests/test_header_parser.py"
    before = {source: source.read_bytes(), test_file: test_file.read_bytes()}
    fake_key = "agent-reproduction-api-key"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    (workspace / "conftest.py").write_text(
        "import os\n\n"
        "def pytest_sessionstart(session):\n"
        "    assert 'OPENAI_API_KEY' not in os.environ\n",
        encoding="utf-8",
    )

    def reject_openai_construction(*args: object, **kwargs: object) -> object:
        raise AssertionError("scripted reproduction workflow attempted an OpenAI request")

    monkeypatch.setattr(openai_agent_module, "OpenAI", reject_openai_construction)
    model = FixtureReproductionModel()

    result = run_reproduction_from_paths(
        task_path=repository_root / "examples/reproduction/empty-header-bug.yaml",
        workspace_root=workspace,
        model=model,
        max_steps=6,
    )

    assert result.state.workflow is AgentWorkflow.REPRODUCTION
    assert result.state.phase is AgentPhase.FINISHED
    assert result.state.step_count == 6
    assert model.call_count == 6
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.evidence.exit_code == 1
    assert attempt.verdict.status is ReproductionStatus.REPRODUCED
    assert len(result.state.reproduction_observations) == 1
    public_rendered = repr(result.state.model_dump())
    assert "target-test-name" not in public_rendered
    assert "target-assertion" not in public_rendered
    assert "expected-test-count" not in public_rendered
    assert "module-import-error" not in public_rendered
    assert "pytest-collection-error" not in public_rendered
    assert "expected_exit_codes" not in public_rendered
    prompts = "\n".join(model.prompt_renderings)
    assert "target-test-name" not in prompts
    assert "module-import-error" not in prompts
    assert "expected_exit_codes" not in prompts
    assert "matched_required_fragment_ids" not in prompts
    assert "missing_required_fragment_ids" not in prompts
    assert "forbidden_fragment_ids_found" not in prompts
    assert result.state.terminal_summary is not None
    assert result.state.terminal_summary == (
        "The reported behavior was reproduced. No patch was generated or verified."
    )
    assert "fixed" not in result.state.terminal_summary
    assert "patch was generated" in result.state.terminal_summary
    assert fake_key not in public_rendered
    assert {path: path.read_bytes() for path in before} == before
    assert not (workspace / ".pytest_cache").exists()
    assert not any(workspace.rglob("__pycache__"))
    assert "patch was applied" not in repr(result.model_dump())
