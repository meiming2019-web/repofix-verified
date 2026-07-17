"""Integration test for structured model decisions with real repository tools."""

import json
from pathlib import Path
from types import SimpleNamespace

from repofix.agent import (
    AgentPhase,
    FinishInvestigationAction,
    IssueUnderstanding,
    ReadFileAction,
    RecordHypothesisAction,
    RepairHypothesis,
    SearchCodeAction,
    UnderstandIssueAction,
    run_read_only_investigation,
)
from repofix.models import AgentDecision, OpenAIResponsesAgentModel
from repofix.models.openai_agent import DEFAULT_MAX_OUTPUT_TOKENS
from repofix.tasks import AgentTaskSpec
from repofix.tools import LocalReadOnlyToolGateway


class SequencedResponses:
    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = decisions
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        turn = len(self.calls)
        self.calls.append(kwargs)
        assert kwargs["text_format"] is AgentDecision
        assert kwargs["store"] is False
        assert kwargs["max_output_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
        assert "previous_response_id" not in kwargs
        assert "conversation" not in kwargs
        assert set(kwargs) == {
            "model",
            "input",
            "text_format",
            "store",
            "max_output_tokens",
        }
        messages = kwargs["input"]
        assert isinstance(messages, list)
        context = json.loads(messages[1]["content"])
        rendered = messages[1]["content"]
        assert "hidden_tests" not in rendered
        assert "gold_patch" not in rendered
        assert "approved_commands" not in rendered
        if turn == 0:
            assert context["state"]["phase"] == "UNDERSTAND"
        elif turn == 1:
            assert context["state"]["phase"] == "EXPLORE"
            assert context["state"]["observations"] == []
        elif turn == 2:
            observations = context["state"]["observations"]
            assert observations[-1]["tool_name"] == "search_code"
            assert "src/parser.py:1:def parse_header" in observations[-1]["output"]
        elif turn == 3:
            observations = context["state"]["observations"]
            assert observations[-1]["tool_name"] == "read_file"
            assert "2:     if not header:" in observations[-1]["output"]
        else:
            assert turn == 4
            assert context["state"]["phase"] == "HYPOTHESIZE"
            assert context["state"]["hypotheses"][0]["hypothesis_id"] == "early-default"
        return SimpleNamespace(output_parsed=self.decisions[turn])


class FakeOpenAIClient:
    def __init__(self, responses: SequencedResponses) -> None:
        self.responses = responses


def test_model_adapter_drives_real_read_only_investigation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    source = workspace / "src/parser.py"
    test_file = workspace / "tests/test_parser.py"
    source.write_text(
        "def parse_header(header):\n    if not header:\n        return DEFAULT\n",
        encoding="utf-8",
    )
    test_file.write_text("def test_empty_header():\n    assert parse_header('')\n", encoding="utf-8")
    before = {source: source.read_bytes(), test_file: test_file.read_bytes()}

    task = AgentTaskSpec.model_validate(
        {
            "task_id": "model-integration",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0123456789abcdef0123456789abcdef01234567",
            "issue_title": "Empty headers return the wrong value",
            "issue_body": "The parser returns DEFAULT for an empty header.",
            "approved_commands": {"unit_tests": {"argv": ["pytest", "-q"]}},
            "allowed_source_paths": ["src", "tests"],
            "timeout_seconds": 300,
        }
    )
    understanding = IssueUnderstanding.model_validate(
        {
            "expected_behavior": "Empty headers retain the configured value.",
            "observed_behavior": "Empty headers return DEFAULT.",
            "reproduction_clues": ["The issue identifies the empty-header case."],
            "likely_components": ["src/parser.py"],
            "missing_information": [],
        }
    )
    hypothesis = RepairHypothesis.model_validate(
        {
            "hypothesis_id": "early-default",
            "description": "The empty-header branch returns DEFAULT too early.",
            "supporting_evidence": ["The read observation shows the early return."],
            "contradicting_evidence": [],
            "confidence": 0.9,
            "status": "supported",
        }
    )
    responses = SequencedResponses(
        [
            AgentDecision(
                action=UnderstandIssueAction(
                    kind="understand_issue", understanding=understanding
                )
            ),
            AgentDecision(
                action=SearchCodeAction(
                    kind="search_code", query="parse_header", file_glob="*.py"
                )
            ),
            AgentDecision(
                action=ReadFileAction(
                    kind="read_file", path="src/parser.py", start_line=1, end_line=3
                )
            ),
            AgentDecision(
                action=RecordHypothesisAction(
                    kind="record_hypothesis", hypothesis=hypothesis
                )
            ),
            AgentDecision(
                action=FinishInvestigationAction(
                    kind="finish_investigation",
                    summary="Read-only investigation identified the likely faulty branch.",
                )
            ),
        ]
    )
    model = OpenAIResponsesAgentModel(
        model="configured-integration-model",
        client=FakeOpenAIClient(responses),  # type: ignore[arg-type]
    )
    tools = LocalReadOnlyToolGateway(
        workspace_root=workspace, allowed_source_paths=task.allowed_source_paths
    )

    state = run_read_only_investigation(task=task, model=model, tools=tools)

    assert state.phase is AgentPhase.FINISHED
    assert state.step_count == 5
    assert [observation.tool_name for observation in state.observations] == [
        "search_code",
        "read_file",
    ]
    assert state.hypotheses == (hypothesis,)
    assert len(responses.calls) == 5
    user_messages = [call["input"][1]["content"] for call in responses.calls]
    assert len(set(user_messages)) == 5
    assert {source: source.read_bytes(), test_file: test_file.read_bytes()} == before
