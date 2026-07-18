"""Tests for the OpenAI Responses API agent adapter."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import ValidationError

import repofix.models.openai_agent as openai_agent_module
from repofix.agent import (
    AgentReproductionObservation,
    AgentPhase,
    AgentState,
    AgentWorkflow,
    RunApprovedCommandAction,
    SearchCodeAction,
)
from repofix.execution import CommandTerminationReason
from repofix.agent.prompts import build_investigation_messages
from repofix.models import AgentDecision, ModelExecutionError, OpenAIResponsesAgentModel
from repofix.models.openai_agent import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_TIMEOUT_SECONDS,
    InvestigationAgentDecision,
    ReproductionPostAttemptDecision,
    ReproductionPreAttemptDecision,
)
from repofix.reproduction import ReproductionStatus
from repofix.tasks import AgentTaskSpec


def task_spec() -> AgentTaskSpec:
    return AgentTaskSpec.model_validate(
        {
            "task_id": "model-task",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0123456789abcdef0123456789abcdef01234567",
            "issue_title": "Find the parser defect",
            "issue_body": "The parser returns the wrong value.",
            "approved_commands": {"unit_tests": {"argv": ["pytest", "-q"]}},
            "allowed_source_paths": ["src"],
            "timeout_seconds": 300,
        }
    )


class FakeResponses:
    def __init__(
        self,
        *,
        parsed: object = None,
        error: BaseException | None = None,
        status: str = "completed",
    ) -> None:
        self.parsed = parsed
        self.error = error
        self.status = status
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_parsed=self.parsed, status=self.status)


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def adapter(responses: FakeResponses) -> OpenAIResponsesAgentModel:
    return OpenAIResponsesAgentModel(
        model="configured-test-model",
        client=FakeClient(responses),  # type: ignore[arg-type]
    )


def valid_decision() -> AgentDecision:
    return AgentDecision(
        action=SearchCodeAction(kind="search_code", query="parse_header", file_glob="*.py")
    )


def test_valid_structured_response_returns_contained_action_and_request_shape() -> None:
    responses = FakeResponses(parsed=valid_decision())
    model = adapter(responses)
    task = task_spec()
    state = AgentState.initial("model-task")

    action = model.next_action(task=task, state=state)

    assert action == valid_decision().action
    assert responses.calls == [
        {
            "model": "configured-test-model",
            "input": build_investigation_messages(task=task, state=state),
            "text_format": AgentDecision,
            "store": False,
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        }
    ]
    assert "previous_response_id" not in responses.calls[0]
    assert "conversation" not in responses.calls[0]


def test_agent_decision_is_strict_and_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentDecision.model_validate(
            {
                "action": {"kind": "search_code", "query": "parse_header"},
                "reasoning": "private",
            }
        )


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_real_sdk_request_uses_compatible_action_schema_and_parses_response() -> None:
    request_bodies: list[dict[str, object]] = []
    action_data = {
        "action": {
            "kind": "search_code",
            "query": "parse_header",
            "file_glob": "*.py",
        }
    }

    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://local.test/v1/responses")
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "configured-test-model",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "text": json.dumps(action_data),
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    transport = httpx.MockTransport(handle_request)
    with httpx.Client(transport=transport) as http_client:
        client = OpenAI(
            api_key="test-api-key",
            base_url="http://local.test/v1",
            max_retries=0,
            http_client=http_client,
        )
        model = OpenAIResponsesAgentModel(
            model="configured-test-model",
            client=client,
        )
        action = model.next_action(
            task=task_spec(), state=AgentState.initial("model-task")
        )

    assert action == AgentDecision.model_validate(action_data).action
    assert len(request_bodies) == 1
    body = request_bodies[0]
    schema = body["text"]["format"]["schema"]
    action_schema = schema["properties"]["action"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "anyOf" in action_schema
    assert not _contains_key(schema, "oneOf")
    assert not _contains_key(schema, "discriminator")
    assert body["store"] is False
    assert body["max_output_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
    assert "previous_response_id" not in body
    assert "conversation" not in body

    action_kinds = {
        schema["$defs"][alternative["$ref"].rsplit("/", 1)[-1]]["properties"]["kind"][
            "const"
        ]
        for alternative in action_schema["anyOf"]
    }
    assert action_kinds == {
        "understand_issue",
        "list_files",
        "search_code",
        "read_file",
        "record_hypothesis",
        "finish_investigation",
    }
    assert "RunApprovedCommandAction" not in schema["$defs"]


def test_adapter_parses_valid_command_action_and_rejects_extra_command_fields() -> None:
    decision = ReproductionPreAttemptDecision(
        action=RunApprovedCommandAction(command_id="unit_tests")
    )
    state = AgentState.initial(
        "model-task",
        workflow=AgentWorkflow.REPRODUCTION,
        reproduction_command_id="unit_tests",
    )
    responses = FakeResponses(parsed=decision)
    action = adapter(responses).next_action(
        task=task_spec(), state=state
    )

    assert action == RunApprovedCommandAction(command_id="unit_tests")
    assert responses.calls[0]["text_format"] is ReproductionPreAttemptDecision
    with pytest.raises(ValidationError):
        ReproductionPreAttemptDecision.model_validate(
            {
                "action": {
                    "kind": "run_approved_command",
                    "command_id": "unit_tests",
                    "argv": ["pytest", "-q"],
                }
            }
        )


def test_adapter_selects_post_attempt_schema_after_consumed_command() -> None:
    observation = AgentReproductionObservation(
        command_id="unit_tests",
        termination_reason=CommandTerminationReason.COMPLETED,
        exit_code=0,
        stdout="",
        stderr="",
        stdout_bytes=0,
        stderr_bytes=0,
        had_decode_errors=False,
        status=ReproductionStatus.NOT_REPRODUCED,
    )
    state = AgentState(
        task_id="model-task",
        phase=AgentPhase.EXPLORE,
        issue_understanding=None,
        hypotheses=(),
        observations=(),
        step_count=1,
        terminal_summary=None,
        failure_reason=None,
        workflow=AgentWorkflow.REPRODUCTION,
        reproduction_command_id="unit_tests",
        reproduction_observations=(observation,),
    )
    decision = ReproductionPostAttemptDecision(
        action=SearchCodeAction(kind="search_code", query="more evidence")
    )
    responses = FakeResponses(parsed=decision)

    action = adapter(responses).next_action(task=task_spec(), state=state)

    assert action == decision.action
    assert responses.calls[0]["text_format"] is ReproductionPostAttemptDecision


def test_workflow_specific_decisions_have_compatible_distinct_action_schemas() -> None:
    investigation_schema = InvestigationAgentDecision.model_json_schema()
    pre_attempt_schema = ReproductionPreAttemptDecision.model_json_schema()
    post_attempt_schema = ReproductionPostAttemptDecision.model_json_schema()

    for schema in (investigation_schema, pre_attempt_schema, post_attempt_schema):
        action_schema = schema["properties"]["action"]
        assert "anyOf" in action_schema
        assert not _contains_key(schema, "oneOf")
        assert not _contains_key(schema, "discriminator")

    assert "run_approved_command" not in repr(investigation_schema)
    assert "finish_investigation" in repr(investigation_schema)
    assert "run_approved_command" in repr(pre_attempt_schema)
    assert "finish_investigation" not in repr(pre_attempt_schema)
    assert "run_approved_command" not in repr(post_attempt_schema)
    assert "finish_investigation" not in repr(post_attempt_schema)
    command_schema = pre_attempt_schema["$defs"]["RunApprovedCommandAction"]
    assert set(command_schema["properties"]) == {"kind", "command_id"}
    assert command_schema["additionalProperties"] is False
    with pytest.raises(ValidationError):
        InvestigationAgentDecision.model_validate(
            {"action": {"kind": "run_approved_command", "command_id": "unit_tests"}}
        )
    for decision_type in (
        ReproductionPreAttemptDecision,
        ReproductionPostAttemptDecision,
    ):
        with pytest.raises(ValidationError):
            decision_type.model_validate(
                {"action": {"kind": "finish_investigation", "summary": "Done"}}
            )
    with pytest.raises(ValidationError):
        ReproductionPostAttemptDecision.model_validate(
            {"action": {"kind": "run_approved_command", "command_id": "unit_tests"}}
        )


def test_real_sdk_reproduction_request_uses_command_schema() -> None:
    request_bodies: list[dict[str, object]] = []
    action_data = {
        "action": {"kind": "run_approved_command", "command_id": "unit_tests"}
    }

    def handle_request(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_reproduction",
                "object": "response",
                "created_at": 0,
                "model": "configured-test-model",
                "output": [
                    {
                        "id": "msg_reproduction",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "text": json.dumps(action_data),
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as http_client:
        client = OpenAI(
            api_key="test-api-key",
            base_url="http://local.test/v1",
            max_retries=0,
            http_client=http_client,
        )
        model = OpenAIResponsesAgentModel(model="configured-test-model", client=client)
        action = model.next_action(
            task=task_spec(),
            state=AgentState.initial(
                "model-task",
                workflow=AgentWorkflow.REPRODUCTION,
                reproduction_command_id="unit_tests",
            ),
        )

    assert action == RunApprovedCommandAction(command_id="unit_tests")
    body = request_bodies[0]
    schema = body["text"]["format"]["schema"]
    assert "anyOf" in schema["properties"]["action"]
    assert not _contains_key(schema, "oneOf")
    assert not _contains_key(schema, "discriminator")
    command_schema = schema["$defs"]["RunApprovedCommandAction"]
    assert set(command_schema["properties"]) == {"kind", "command_id"}
    assert body["store"] is False
    assert body["max_output_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
    assert "previous_response_id" not in body
    assert "conversation" not in body


@pytest.mark.parametrize("parsed", [None, object()])
def test_missing_or_unexpected_parsed_output_raises_model_error(parsed: object) -> None:
    responses = FakeResponses(parsed=parsed)
    with pytest.raises(ModelExecutionError, match="structured decision|unexpected parsed"):
        adapter(responses).next_action(
            task=task_spec(), state=AgentState.initial("model-task")
        )

    assert len(responses.calls) == 1


def test_incomplete_response_without_parsed_output_is_not_retried() -> None:
    responses = FakeResponses(parsed=None, status="incomplete")

    with pytest.raises(ModelExecutionError, match="no valid structured decision"):
        adapter(responses).next_action(
            task=task_spec(), state=AgentState.initial("model-task")
        )

    assert len(responses.calls) == 1


def provider_errors() -> list[BaseException]:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(500, request=request)
    return [
        APIConnectionError(message="connection failed", request=request),
        APITimeoutError(request),
        APIStatusError("status failed", response=response, body=None),
    ]


@pytest.mark.parametrize("provider_error", provider_errors())
def test_expected_provider_errors_are_wrapped_with_original_cause(
    provider_error: BaseException,
) -> None:
    with pytest.raises(ModelExecutionError, match="model request failed") as caught:
        adapter(FakeResponses(error=provider_error)).next_action(
            task=task_spec(), state=AgentState.initial("model-task")
        )

    assert caught.value.__cause__ is provider_error


def test_structured_validation_error_is_wrapped_without_sensitive_output() -> None:
    invalid_output_marker = "private-repository-content"
    validation_error: ValidationError | None = None
    try:
        AgentDecision.model_validate(
            {
                "action": {
                    "kind": "read_file",
                    "path": invalid_output_marker,
                    "start_line": 10,
                    "end_line": 1,
                }
            }
        )
    except ValidationError as error:
        validation_error = error
    else:
        raise AssertionError("test input must produce a validation error")

    assert validation_error is not None
    with pytest.raises(ModelExecutionError) as caught:
        adapter(FakeResponses(error=validation_error)).next_action(
            task=task_spec(), state=AgentState.initial("model-task")
        )

    assert str(caught.value) == "OpenAI model returned an invalid structured decision"
    assert caught.value.__cause__ is validation_error
    assert invalid_output_marker not in str(caught.value)
    assert task_spec().issue_body not in str(caught.value)
    assert "test-api-key" not in str(caught.value)


def test_unexpected_programmer_error_propagates() -> None:
    error = AssertionError("adapter invariant failed")

    with pytest.raises(AssertionError, match="adapter invariant failed"):
        adapter(FakeResponses(error=error)).next_action(
            task=task_spec(), state=AgentState.initial("model-task")
        )


def test_unexpected_type_error_propagates() -> None:
    with pytest.raises(TypeError, match="unexpected SDK contract error"):
        adapter(FakeResponses(error=TypeError("unexpected SDK contract error"))).next_action(
            task=task_spec(), state=AgentState.initial("model-task")
        )


@pytest.mark.parametrize("model_name", ["", "   "])
def test_empty_model_name_is_rejected(model_name: str) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        OpenAIResponsesAgentModel(
            model=model_name,
            client=FakeClient(FakeResponses()),  # type: ignore[arg-type]
        )


def test_default_client_disables_retries_and_uses_named_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments: dict[str, object] = {}
    client = FakeClient(FakeResponses())

    def create_client(**kwargs: object) -> FakeClient:
        arguments.update(kwargs)
        return client

    monkeypatch.setattr(openai_agent_module, "OpenAI", create_client)

    OpenAIResponsesAgentModel(model="configured-test-model")

    assert arguments == {
        "max_retries": 0,
        "timeout": DEFAULT_MODEL_TIMEOUT_SECONDS,
    }


def test_public_error_does_not_expose_prompt_or_api_credentials() -> None:
    secret = "sk-secret-must-not-appear"
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    provider_error = APIConnectionError(message=f"connection failed with {secret}", request=request)

    with pytest.raises(ModelExecutionError) as caught:
        adapter(FakeResponses(error=provider_error)).next_action(
            task=task_spec(), state=AgentState.initial("model-task")
        )

    message = str(caught.value)
    assert secret not in message
    assert task_spec().issue_body not in message
    assert "parse_header" not in message
