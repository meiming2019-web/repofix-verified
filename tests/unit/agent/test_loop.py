"""Tests for the deterministic read-only investigation loop."""

from collections.abc import Callable
import hashlib

import pytest

from repofix.agent import (
    AgentAction,
    AgentPhase,
    AgentProtocolError,
    AgentState,
    FinishInvestigationAction,
    IssueUnderstanding,
    ListFilesAction,
    ReadFileAction,
    ReadFileResult,
    RecordHypothesisAction,
    RepairHypothesis,
    SearchCodeAction,
    ToolExecutionError,
    UnderstandIssueAction,
    run_read_only_investigation,
)
from repofix.tasks import AgentTaskSpec


StateAssertion = Callable[[AgentState], None]


def task_spec() -> AgentTaskSpec:
    return AgentTaskSpec.model_validate(
        {
            "task_id": "task-001",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0123456789abcdef0123456789abcdef01234567",
            "issue_title": "Parser returns the wrong value",
            "issue_body": "An empty header causes the configured value to be discarded.",
            "approved_commands": {"unit_tests": {"argv": ["pytest", "-q"]}},
            "allowed_source_paths": ["src/repofix", "tests/unit"],
            "timeout_seconds": 300,
        }
    )


def understanding() -> IssueUnderstanding:
    return IssueUnderstanding.model_validate(
        {
            "expected_behavior": "The parser returns the configured value.",
            "observed_behavior": "The parser returns a default value.",
            "reproduction_clues": ["An empty header triggers the failure."],
            "likely_components": ["src/repofix/parser.py"],
            "missing_information": [],
        }
    )


def hypothesis(identifier: str = "hypothesis-1") -> RepairHypothesis:
    return RepairHypothesis.model_validate(
        {
            "hypothesis_id": identifier,
            "description": "The empty-header branch discards the configured value.",
            "supporting_evidence": ["Search and source output identify the branch."],
            "contradicting_evidence": [],
            "confidence": 0.8,
            "status": "supported",
        }
    )


class ScriptedModel:
    def __init__(
        self, actions: list[AgentAction], assertions: list[StateAssertion] | None = None
    ) -> None:
        self.actions = actions
        self.assertions = assertions or [lambda state: None for _ in actions]
        self.call_count = 0
        self.received_phases: list[AgentPhase] = []

    def next_action(self, *, task: AgentTaskSpec, state: AgentState) -> AgentAction:
        assert task.task_id == state.task_id
        index = self.call_count
        self.call_count += 1
        self.received_phases.append(state.phase)
        self.assertions[index](state)
        return self.actions[index]


class ScriptedTools:
    def __init__(self, *, search_error: Exception | None = None) -> None:
        self.search_error = search_error
        self.calls: list[tuple[str, dict[str, object]]] = []

    def list_files(self, path: str) -> str:
        self.calls.append(("list_files", {"path": path}))
        return "src/repofix/parser.py"

    def search_code(self, query: str, file_glob: str | None = None) -> str:
        self.calls.append(("search_code", {"query": query, "file_glob": file_glob}))
        if self.search_error is not None:
            raise self.search_error
        return "src/repofix/parser.py:14:def parse_header"

    def read_file(self, path: str, start_line: int, end_line: int) -> str:
        self.calls.append(
            (
                "read_file",
                {"path": path, "start_line": start_line, "end_line": end_line},
            )
        )
        return "14: def parse_header(header):\n15:     return DEFAULT"

    def read_file_with_metadata(self, path: str, start_line: int, end_line: int) -> ReadFileResult:
        return ReadFileResult(
            output=self.read_file(path, start_line, end_line),
            full_file_sha256=hashlib.sha256(b"complete fake source").hexdigest(),
        )


def test_complete_deterministic_trajectory_feeds_observations_back() -> None:
    actions: list[AgentAction] = [
        UnderstandIssueAction(kind="understand_issue", understanding=understanding()),
        SearchCodeAction(kind="search_code", query="parse_header", file_glob="*.py"),
        ReadFileAction(
            kind="read_file", path="src/repofix/parser.py", start_line=10, end_line=20
        ),
        RecordHypothesisAction(kind="record_hypothesis", hypothesis=hypothesis()),
        FinishInvestigationAction(
            kind="finish_investigation",
            summary="The investigation identified a likely faulty branch for later repair.",
        ),
    ]

    def initial_assertion(state: AgentState) -> None:
        assert state == AgentState.initial("task-001")

    def understood_assertion(state: AgentState) -> None:
        assert state.issue_understanding == understanding()
        assert state.observations == ()

    def search_observed_assertion(state: AgentState) -> None:
        assert state.observations[-1].tool_name == "search_code"
        assert "parse_header" in state.observations[-1].output

    def file_observed_assertion(state: AgentState) -> None:
        assert len(state.observations) == 2
        assert state.observations[-1].tool_name == "read_file"
        assert "return DEFAULT" in state.observations[-1].output
        assert (
            state.observations[-1].full_file_sha256
            == hashlib.sha256(b"complete fake source").hexdigest()
        )

    def hypothesis_assertion(state: AgentState) -> None:
        assert state.hypotheses == (hypothesis(),)
        assert len(state.observations) == 2

    model = ScriptedModel(
        actions,
        [
            initial_assertion,
            understood_assertion,
            search_observed_assertion,
            file_observed_assertion,
            hypothesis_assertion,
        ],
    )
    tools = ScriptedTools()

    state = run_read_only_investigation(task=task_spec(), model=model, tools=tools)

    assert state.phase is AgentPhase.FINISHED
    assert state.step_count == 5
    assert state.terminal_summary == actions[-1].summary
    assert state.failure_reason is None
    assert model.received_phases == [
        AgentPhase.UNDERSTAND,
        AgentPhase.EXPLORE,
        AgentPhase.EXPLORE,
        AgentPhase.EXPLORE,
        AgentPhase.HYPOTHESIZE,
    ]
    assert tools.calls == [
        ("search_code", {"query": "parse_header", "file_glob": "*.py"}),
        (
            "read_file",
            {"path": "src/repofix/parser.py", "start_line": 10, "end_line": 20},
        ),
    ]
    assert state.observations[0].arguments == {
        "query": "parse_header",
        "file_glob": "*.py",
    }
    assert state.observations[0].success is True
    assert state.observations[0].error is None
    assert state.observations[0].full_file_sha256 is None


def test_failed_tool_call_is_observed_and_does_not_crash_loop() -> None:
    def assert_failure_observed(state: AgentState) -> None:
        observation = state.observations[-1]
        assert observation.success is False
        assert observation.output == ""
        assert observation.error is not None
        assert "search backend unavailable" in observation.error

    model = ScriptedModel(
        [
            UnderstandIssueAction(kind="understand_issue", understanding=understanding()),
            SearchCodeAction(kind="search_code", query="parse_header", file_glob=None),
            RecordHypothesisAction(kind="record_hypothesis", hypothesis=hypothesis()),
            FinishInvestigationAction(
                kind="finish_investigation", summary="Investigation ended with a tool failure."
            ),
        ],
        [lambda state: None, lambda state: None, assert_failure_observed, lambda state: None],
    )

    state = run_read_only_investigation(
        task=task_spec(),
        model=model,
        tools=ScriptedTools(search_error=ToolExecutionError("search backend unavailable")),
    )

    assert state.phase is AgentPhase.FINISHED
    assert state.step_count == 4
    assert state.observations[0].success is False


def test_rejects_invalid_action_in_understand_phase() -> None:
    model = ScriptedModel([ListFilesAction(kind="list_files", path="src")])

    with pytest.raises(AgentProtocolError, match="UNDERSTAND"):
        run_read_only_investigation(task=task_spec(), model=model, tools=ScriptedTools())


def test_rejects_premature_finish_in_understand_phase() -> None:
    model = ScriptedModel(
        [FinishInvestigationAction(kind="finish_investigation", summary="Too early")]
    )

    with pytest.raises(AgentProtocolError, match="UNDERSTAND"):
        run_read_only_investigation(task=task_spec(), model=model, tools=ScriptedTools())


def test_rejects_finish_outside_hypothesize_phase() -> None:
    model = ScriptedModel(
        [
            UnderstandIssueAction(kind="understand_issue", understanding=understanding()),
            FinishInvestigationAction(kind="finish_investigation", summary="Still too early"),
        ]
    )

    with pytest.raises(AgentProtocolError, match="EXPLORE"):
        run_read_only_investigation(task=task_spec(), model=model, tools=ScriptedTools())


def test_assertion_error_from_tool_propagates() -> None:
    model = ScriptedModel(
        [
            UnderstandIssueAction(kind="understand_issue", understanding=understanding()),
            SearchCodeAction(kind="search_code", query="parse_header", file_glob=None),
        ]
    )

    with pytest.raises(AssertionError, match="tool invariant failed"):
        run_read_only_investigation(
            task=task_spec(),
            model=model,
            tools=ScriptedTools(search_error=AssertionError("tool invariant failed")),
        )

    assert model.call_count == 2


def test_plain_runtime_error_from_tool_propagates() -> None:
    model = ScriptedModel(
        [
            UnderstandIssueAction(kind="understand_issue", understanding=understanding()),
            SearchCodeAction(kind="search_code", query="parse_header", file_glob=None),
        ]
    )

    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        run_read_only_investigation(
            task=task_spec(),
            model=model,
            tools=ScriptedTools(search_error=RuntimeError("unexpected implementation failure")),
        )

    assert model.call_count == 2


def test_rejects_finish_when_no_tool_observation_exists() -> None:
    model = ScriptedModel(
        [
            UnderstandIssueAction(kind="understand_issue", understanding=understanding()),
            RecordHypothesisAction(kind="record_hypothesis", hypothesis=hypothesis()),
            FinishInvestigationAction(kind="finish_investigation", summary="No evidence gathered"),
        ]
    )

    with pytest.raises(AgentProtocolError, match="tool observation"):
        run_read_only_investigation(task=task_spec(), model=model, tools=ScriptedTools())


def test_tool_action_after_hypothesis_returns_to_explore() -> None:
    def assert_returned_to_explore(state: AgentState) -> None:
        assert state.phase is AgentPhase.EXPLORE
        assert len(state.hypotheses) == 1
        assert len(state.observations) == 2

    model = ScriptedModel(
        [
            UnderstandIssueAction(kind="understand_issue", understanding=understanding()),
            SearchCodeAction(kind="search_code", query="parse_header", file_glob=None),
            RecordHypothesisAction(kind="record_hypothesis", hypothesis=hypothesis()),
            ReadFileAction(kind="read_file", path="src/parser.py", start_line=1, end_line=5),
            RecordHypothesisAction(
                kind="record_hypothesis", hypothesis=hypothesis("hypothesis-2")
            ),
            FinishInvestigationAction(kind="finish_investigation", summary="Investigation ended."),
        ],
        [
            lambda state: None,
            lambda state: None,
            lambda state: None,
            lambda state: None,
            assert_returned_to_explore,
            lambda state: None,
        ],
    )

    state = run_read_only_investigation(task=task_spec(), model=model, tools=ScriptedTools())

    assert state.phase is AgentPhase.FINISHED
    assert len(state.hypotheses) == 2


def test_step_budget_returns_failed_without_extra_model_request() -> None:
    model = ScriptedModel(
        [
            UnderstandIssueAction(kind="understand_issue", understanding=understanding()),
            SearchCodeAction(kind="search_code", query="parse_header", file_glob=None),
            ListFilesAction(kind="list_files", path="src"),
            FinishInvestigationAction(kind="finish_investigation", summary="Must not be requested"),
        ]
    )

    state = run_read_only_investigation(
        task=task_spec(), model=model, tools=ScriptedTools(), max_steps=3
    )

    assert state.phase is AgentPhase.FAILED
    assert state.step_count == 3
    assert state.failure_reason is not None
    assert "3-step budget" in state.failure_reason
    assert state.terminal_summary is None
    assert model.call_count == 3


@pytest.mark.parametrize("max_steps", [True, 0, -1, 1.5, "2"])
def test_rejects_invalid_strict_step_budget(max_steps: object) -> None:
    with pytest.raises(ValueError, match="strict positive integer"):
        run_read_only_investigation(
            task=task_spec(),
            model=ScriptedModel([]),
            tools=ScriptedTools(),
            max_steps=max_steps,  # type: ignore[arg-type]
        )


def test_final_state_contains_no_resolution_or_evaluator_data() -> None:
    model = ScriptedModel(
        [
            UnderstandIssueAction(kind="understand_issue", understanding=understanding()),
            SearchCodeAction(kind="search_code", query="parse_header", file_glob=None),
            RecordHypothesisAction(kind="record_hypothesis", hypothesis=hypothesis()),
            FinishInvestigationAction(
                kind="finish_investigation",
                summary="Read-only investigation complete; the hypothesis still requires repair and verification.",
            ),
        ]
    )

    state = run_read_only_investigation(task=task_spec(), model=model, tools=ScriptedTools())
    serialized = state.model_dump()
    rendered = repr(serialized)

    assert "resolved" not in serialized
    assert "patch" not in serialized
    assert "approved_commands" not in serialized
    assert "hidden_tests" not in rendered
    assert "gold_patch" not in rendered
    assert "command execution" not in rendered
