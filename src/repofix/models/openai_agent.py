"""OpenAI Responses API adapter for read-only investigations.

Intended manual construction::

    model = OpenAIResponsesAgentModel(model="configured-model")
    tools = LocalReadOnlyToolGateway(...)
    state = run_read_only_investigation(task=task, model=model, tools=tools)

The OpenAI SDK obtains credentials through its standard environment behavior.
"""

from typing import Any, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.responses import ResponseInputParam
from pydantic import BaseModel, ConfigDict, ValidationError

from repofix.agent.actions import (
    AgentAction,
    InvestigationAgentAction,
    ReproductionPostAttemptAgentAction,
    ReproductionPreAttemptAgentAction,
)
from repofix.agent.prompts import build_investigation_messages
from repofix.agent.state import AgentState, AgentWorkflow
from repofix.tasks import AgentTaskSpec


DEFAULT_MODEL_TIMEOUT_SECONDS = 60.0
"""Maximum time allowed for one synchronous model decision."""

DEFAULT_MAX_OUTPUT_TOKENS = 2_000
"""Maximum model output tokens allowed for one structured decision."""


def _use_openai_compatible_action_union(schema: dict[str, Any]) -> None:
    """Replace only the model-facing discriminated action union keywords."""
    action_schema = schema["properties"]["action"]
    action_schema["anyOf"] = action_schema.pop("oneOf")
    action_schema.pop("discriminator")


class _AgentDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        json_schema_extra=_use_openai_compatible_action_union,
    )


class InvestigationAgentDecision(_AgentDecision):
    """Strict structured-output envelope for an investigation action."""

    action: InvestigationAgentAction


class ReproductionPreAttemptDecision(_AgentDecision):
    """Strict envelope for reproduction actions before command execution."""

    action: ReproductionPreAttemptAgentAction


class ReproductionPostAttemptDecision(_AgentDecision):
    """Strict envelope for reproduction actions after command execution."""

    action: ReproductionPostAttemptAgentAction


AgentDecision = InvestigationAgentDecision


class ModelExecutionError(RuntimeError):
    """Raised when a model decision cannot be obtained safely."""


class OpenAIResponsesAgentModel:
    """Select one validated action using the synchronous Responses API."""

    def __init__(self, *, model: str, client: OpenAI | None = None) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model name must be a nonempty string")
        self._model = model
        self._client = (
            client
            if client is not None
            else OpenAI(max_retries=0, timeout=DEFAULT_MODEL_TIMEOUT_SECONDS)
        )

    def next_action(self, *, task: AgentTaskSpec, state: AgentState) -> AgentAction:
        """Build fresh state-derived messages and request one structured action."""
        messages = build_investigation_messages(task=task, state=state)
        decision_type: type[_AgentDecision]
        if state.workflow is AgentWorkflow.INVESTIGATION:
            decision_type = InvestigationAgentDecision
        elif state.reproduction_observations:
            decision_type = ReproductionPostAttemptDecision
        else:
            decision_type = ReproductionPreAttemptDecision
        try:
            response = self._client.responses.parse(
                model=self._model,
                input=cast(ResponseInputParam, messages),
                text_format=decision_type,
                store=False,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            )
        except ValidationError as error:
            raise ModelExecutionError(
                "OpenAI model returned an invalid structured decision"
            ) from error
        except (APIConnectionError, APITimeoutError, APIStatusError) as error:
            raise ModelExecutionError("OpenAI model request failed") from error

        decision = response.output_parsed
        if decision is None:
            raise ModelExecutionError("OpenAI model returned no valid structured decision")
        if not isinstance(decision, decision_type):
            raise ModelExecutionError("OpenAI model returned an unexpected parsed result")
        return decision.action
