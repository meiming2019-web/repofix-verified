"""Integration of the OpenAI adapter with the reproduction workflow."""

from types import SimpleNamespace

from repofix.agent import (
    AgentPhase,
    IssueUnderstanding,
    RecordHypothesisAction,
    RepairHypothesis,
    RunApprovedCommandAction,
    SearchCodeAction,
    UnderstandIssueAction,
    run_reproduction_agent_loop,
)
from repofix.execution import (
    ApprovedCommandExecutionResult,
    CommandTerminationReason,
)
from repofix.models import OpenAIResponsesAgentModel
from repofix.models.openai_agent import ReproductionPreAttemptDecision
from repofix.reproduction import (
    ReproductionExpectation,
    ReproductionOutputFragment,
    ReproductionOutputStream,
    ReproductionStatus,
)
from repofix.tasks import AgentTaskSpec


class SequencedResponses:
    def __init__(self, decisions: list[ReproductionPreAttemptDecision]) -> None:
        self.decisions = decisions
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.decisions[len(self.calls) - 1])


class FakeClient:
    def __init__(self, responses: SequencedResponses) -> None:
        self.responses = responses


class FakeTools:
    def list_files(self, path: str) -> str:
        return "src/parser.py"

    def search_code(self, query: str, file_glob: str | None = None) -> str:
        return "src/parser.py:2:return WRONG"

    def read_file(self, path: str, start_line: int, end_line: int) -> str:
        return "2: return WRONG"


class FakeCommandGateway:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, command_id: str) -> ApprovedCommandExecutionResult:
        self.calls.append(command_id)
        output = "MODEL_TARGET_OUTPUT\n"
        return ApprovedCommandExecutionResult(
            command_id=command_id,
            argv=("never-render-executable", "--private-argv"),
            termination_reason=CommandTerminationReason.COMPLETED,
            exit_code=1,
            stdout=output,
            stderr="",
            stdout_bytes=len(output.encode("utf-8")),
            stderr_bytes=0,
            had_decode_errors=False,
        )


def test_model_adapter_requests_exact_command_and_receives_only_sanitized_result() -> None:
    task = AgentTaskSpec.model_validate(
        {
            "task_id": "model-reproduction",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0123456789abcdef0123456789abcdef01234567",
            "issue_title": "Target behavior fails",
            "issue_body": "The parser returns the wrong value.",
            "approved_commands": {
                "unit_tests": {
                    "argv": ["never-render-executable", "--private-argv"]
                }
            },
            "allowed_source_paths": ["src"],
            "timeout_seconds": 300,
        }
    )
    expectation = ReproductionExpectation(
        command_id="unit_tests",
        expected_exit_codes=(1,),
        required_fragments=(
            ReproductionOutputFragment(
                fragment_id="private-target-fragment-id",
                stream=ReproductionOutputStream.COMBINED,
                text="MODEL_TARGET_OUTPUT",
            ),
        ),
        forbidden_fragments=(
            ReproductionOutputFragment(
                fragment_id="private-forbidden-fragment-id",
                stream=ReproductionOutputStream.COMBINED,
                text="PRIVATE_FORBIDDEN_OUTPUT",
            ),
        ),
    )
    understanding = IssueUnderstanding(
        expected_behavior="The parser returns the configured value.",
        observed_behavior="The parser returns the wrong value.",
        reproduction_clues=("The issue identifies the target case.",),
        likely_components=("src/parser.py",),
        missing_information=(),
    )
    hypothesis = RepairHypothesis(
        hypothesis_id="wrong-return",
        description="The target branch returns the wrong value.",
        supporting_evidence=("Search output identifies the return.",),
        contradicting_evidence=(),
        confidence=0.9,
        status="supported",
    )
    responses = SequencedResponses(
        [
            ReproductionPreAttemptDecision(
                action=UnderstandIssueAction(
                    kind="understand_issue", understanding=understanding
                )
            ),
            ReproductionPreAttemptDecision(
                action=SearchCodeAction(
                    kind="search_code", query="return WRONG", file_glob="*.py"
                )
            ),
            ReproductionPreAttemptDecision(
                action=RecordHypothesisAction(
                    kind="record_hypothesis", hypothesis=hypothesis
                )
            ),
            ReproductionPreAttemptDecision(
                action=RunApprovedCommandAction(command_id="unit_tests")
            ),
        ]
    )
    model = OpenAIResponsesAgentModel(
        model="configured-reproduction-model",
        client=FakeClient(responses),  # type: ignore[arg-type]
    )
    gateway = FakeCommandGateway()

    result = run_reproduction_agent_loop(
        task=task,
        expectation=expectation,
        model=model,
        tools=FakeTools(),
        command_gateway=gateway,
        max_steps=4,
    )

    assert result.state.phase is AgentPhase.FINISHED
    assert result.state.terminal_summary == (
        "The reported behavior was reproduced. No patch was generated or verified."
    )
    assert result.attempts[0].verdict.status is ReproductionStatus.REPRODUCED
    assert gateway.calls == ["unit_tests"]
    assert len(responses.calls) == 4
    request_renderings = [repr(call["input"]) for call in responses.calls]
    for rendered in request_renderings:
        assert "never-render-executable" not in rendered
        assert "--private-argv" not in rendered
        assert "private-target-fragment-id" not in rendered
        assert "private-forbidden-fragment-id" not in rendered
        assert "PRIVATE_FORBIDDEN_OUTPUT" not in rendered
        assert "expected_exit_codes" not in rendered
        assert "matched_required_fragment_ids" not in rendered
        assert "missing_required_fragment_ids" not in rendered
        assert "forbidden_fragment_ids_found" not in rendered
    assert all("MODEL_TARGET_OUTPUT" not in rendered for rendered in request_renderings)
    assert all(call["store"] is False for call in responses.calls)
    assert all(
        call["text_format"] is ReproductionPreAttemptDecision
        for call in responses.calls
    )
    assert all("previous_response_id" not in call for call in responses.calls)
    assert all("conversation" not in call for call in responses.calls)
