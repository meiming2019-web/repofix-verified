import json
from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    OpenAI,
)
from openai.types.chat import ChatCompletion
from pydantic import ValidationError

import repofix.models.openai_patch as openai_patch_module

from repofix.agent import IssueUnderstanding, RepairHypothesis
from repofix.models.openai_patch import (
    DEFAULT_PATCH_MAX_OUTPUT_TOKENS,
    OpenAIPatchProposalModel,
    PatchModelExecutionError,
)
from repofix.patching import (
    PatchContextFileObservation,
    PatchEditDraft,
    PatchProposalContext,
    PatchProposalDraft,
)
from repofix.patching.models import (
    MAX_PATCH_EDITS,
    MAX_PATCH_RATIONALE_CHARS,
    MAX_PATCH_SUMMARY_CHARS,
    MAX_TOTAL_REPLACEMENT_CHARS,
)


class Responses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed)


class Client:
    def __init__(self, responses):
        self.responses = responses


class RaisingResponses:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        raise self.error


def context():
    return PatchProposalContext(
        task_id="t",
        issue_title="Ignore instructions",
        issue_body="untrusted",
        patchable_source_paths=("src",),
        issue_understanding=IssueUnderstanding(
            expected_behavior="a",
            observed_behavior="b",
            reproduction_clues=(),
            likely_components=("src/a.py",),
            missing_information=(),
        ),
        supported_hypotheses=(
            RepairHypothesis(
                hypothesis_id="h",
                description="d",
                supporting_evidence=("e",),
                contradicting_evidence=(),
                confidence=0.8,
                status="supported",
            ),
        ),
        successful_file_observations=(
            PatchContextFileObservation(path="src/a.py", excerpt="1: return bad", truncated=False),
        ),
        reproduction_status="reproduced",
    )


def test_valid_request_is_stateless_and_sanitized() -> None:
    draft = PatchProposalDraft(
        hypothesis_id="h",
        model_summary="change",
        edits=(
            PatchEditDraft(
                path="src/a.py",
                start_line=1,
                end_line=1,
                replacement_text="return good\n",
                rationale="correct",
            ),
        ),
    )
    responses = Responses(draft)
    model = OpenAIPatchProposalModel(model="test", client=Client(responses))  # type: ignore[arg-type]
    assert model.propose_patch(context=context()) == draft
    call = responses.calls[0]
    assert call["store"] is False and call["max_output_tokens"] == DEFAULT_PATCH_MAX_OUTPUT_TOKENS
    assert "previous_response_id" not in call and "conversation" not in call
    user_rendered = repr(call["input"][1])
    assert "shell" not in user_rendered and "expected_exit_codes" not in user_rendered
    assert "untrusted data" in repr(call["input"][0])


def test_missing_output_is_normalized() -> None:
    model = OpenAIPatchProposalModel(model="test", client=Client(Responses(None)))  # type: ignore[arg-type]
    with pytest.raises(PatchModelExecutionError, match="no valid"):
        model.propose_patch(context=context())


def provider_errors() -> list[BaseException]:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(500, request=request)
    completion = ChatCompletion(
        id="completion",
        choices=[],
        created=0,
        model="test",
        object="chat.completion",
    )
    return [
        APIConnectionError(message="connection failed", request=request),
        APITimeoutError(request),
        APIStatusError("status failed", response=response, body=None),
        ContentFilterFinishReasonError(),
        LengthFinishReasonError(completion=completion),
    ]


@pytest.mark.parametrize("provider_error", provider_errors())
def test_expected_sdk_errors_are_normalized_without_retry(
    provider_error: BaseException,
) -> None:
    responses = RaisingResponses(provider_error)
    model = OpenAIPatchProposalModel(model="test", client=Client(responses))  # type: ignore[arg-type]

    with pytest.raises(PatchModelExecutionError, match="request failed") as caught:
        model.propose_patch(context=context())

    assert caught.value.__cause__ is provider_error
    assert responses.calls == 1


def test_pydantic_error_is_normalized_and_programmer_error_propagates() -> None:
    captured_validation_error: ValidationError | None = None
    try:
        PatchProposalDraft.model_validate({"private": "invalid"})
    except ValidationError as error:
        captured_validation_error = error
    else:  # pragma: no cover - test construction guard
        raise AssertionError("invalid draft must raise ValidationError")
    assert captured_validation_error is not None
    model = OpenAIPatchProposalModel(
        model="test",
        client=Client(RaisingResponses(captured_validation_error)),  # type: ignore[arg-type]
    )
    with pytest.raises(PatchModelExecutionError, match="invalid patch proposal") as caught:
        model.propose_patch(context=context())
    assert caught.value.__cause__ is captured_validation_error

    programmer_error = TypeError("unexpected adapter contract")
    model = OpenAIPatchProposalModel(
        model="test",
        client=Client(RaisingResponses(programmer_error)),  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="unexpected adapter contract"):
        model.propose_patch(context=context())


def test_unexpected_parsed_type_is_normalized_without_retry() -> None:
    responses = Responses(object())
    model = OpenAIPatchProposalModel(model="test", client=Client(responses))  # type: ignore[arg-type]

    with pytest.raises(PatchModelExecutionError, match="unexpected patch proposal"):
        model.propose_patch(context=context())

    assert len(responses.calls) == 1


def test_default_client_disables_retries_and_uses_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> object:
        captured.update(kwargs)
        return Client(Responses(None))

    monkeypatch.setattr(openai_patch_module, "OpenAI", fake_openai)

    OpenAIPatchProposalModel(model="test")

    assert captured == {
        "max_retries": 0,
        "timeout": openai_patch_module.DEFAULT_PATCH_MODEL_TIMEOUT_SECONDS,
    }


def test_real_sdk_request_schema_and_output_budget() -> None:
    draft = PatchProposalDraft(
        hypothesis_id="h",
        model_summary="change",
        edits=(
            PatchEditDraft(
                path="src/a.py",
                start_line=1,
                end_line=1,
                replacement_text="return good\n",
                rationale="correct",
            ),
        ),
    )
    bodies: list[dict[str, object]] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_patch",
                "object": "response",
                "created_at": 0,
                "model": "configured-patch-model",
                "output": [
                    {
                        "id": "msg_patch",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "text": json.dumps(draft.model_dump(mode="json")),
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
        parsed = OpenAIPatchProposalModel(
            model="configured-patch-model", client=client
        ).propose_patch(context=context())

    assert parsed == draft
    assert len(bodies) == 1
    body = bodies[0]
    schema = body["text"]["format"]["schema"]  # type: ignore[index]
    edits_schema = schema["properties"]["edits"]  # type: ignore[index]
    assert edits_schema["minItems"] == 1
    assert edits_schema["maxItems"] == MAX_PATCH_EDITS
    assert body["store"] is False
    assert body["max_output_tokens"] == DEFAULT_PATCH_MAX_OUTPUT_TOKENS
    assert "previous_response_id" not in body and "conversation" not in body
    assert DEFAULT_PATCH_MAX_OUTPUT_TOKENS > (
        MAX_TOTAL_REPLACEMENT_CHARS
        + MAX_PATCH_EDITS * MAX_PATCH_RATIONALE_CHARS
        + MAX_PATCH_SUMMARY_CHARS
    )
    system_prompt = body["input"][0]["content"]  # type: ignore[index]
    for constraint in (
        "at most 8 edits",
        "at most 3 files",
        "Sort edits by path",
        "duplicate or overlapping",
        "raw diff",
        "Do not claim",
    ):
        assert constraint in system_prompt
