"""Strict state models for the read-only investigation agent."""

from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentPhase(str, Enum):
    """Phases of a read-only agent investigation."""

    UNDERSTAND = "UNDERSTAND"
    EXPLORE = "EXPLORE"
    HYPOTHESIZE = "HYPOTHESIZE"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _normalize_sequence(value: object, field_name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    return tuple(value)


class IssueUnderstanding(_StrictFrozenModel):
    """Structured understanding of the reported issue."""

    expected_behavior: str
    observed_behavior: str
    reproduction_clues: tuple[str, ...]
    likely_components: tuple[str, ...]
    missing_information: tuple[str, ...]

    @field_validator("reproduction_clues", "likely_components", "missing_information", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object) -> tuple[object, ...]:
        return _normalize_sequence(value, "issue-understanding sequence")

    @field_validator("expected_behavior", "observed_behavior")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("behavior descriptions must not be empty")
        return value


class RepairHypothesis(_StrictFrozenModel):
    """An explicit, evidence-linked repair hypothesis."""

    hypothesis_id: str
    description: str
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal["unverified", "supported", "rejected"]

    @field_validator("supporting_evidence", "contradicting_evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: object) -> tuple[object, ...]:
        return _normalize_sequence(value, "evidence")

    @field_validator("hypothesis_id", "description")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("hypothesis ID and description must not be empty")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def reject_boolean_confidence(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("confidence must not be a boolean")
        return value


class ToolObservation(_StrictFrozenModel):
    """Public result of one read-only tool action."""

    step_index: int = Field(ge=0)
    tool_name: str
    arguments: dict[str, object]
    success: bool
    output: str
    error: str | None

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool name must not be empty")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.success and self.error is not None:
            raise ValueError("successful tool observations must not contain an error")
        if not self.success and (self.error is None or not self.error.strip()):
            raise ValueError("failed tool observations must contain an error")
        return self


class AgentState(_StrictFrozenModel):
    """Complete public state of a read-only investigation."""

    task_id: str
    phase: AgentPhase
    issue_understanding: IssueUnderstanding | None
    hypotheses: tuple[RepairHypothesis, ...]
    observations: tuple[ToolObservation, ...]
    step_count: int = Field(ge=0)
    terminal_summary: str | None
    failure_reason: str | None

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task ID must not be empty")
        return value

    @model_validator(mode="after")
    def validate_phase_result(self) -> Self:
        nonterminal_phases = {
            AgentPhase.UNDERSTAND,
            AgentPhase.EXPLORE,
            AgentPhase.HYPOTHESIZE,
        }
        if self.phase in nonterminal_phases:
            if self.terminal_summary is not None:
                raise ValueError("nonterminal states must not contain a terminal summary")
            if self.failure_reason is not None:
                raise ValueError("nonterminal states must not contain a failure reason")
        elif self.phase is AgentPhase.FINISHED:
            if self.terminal_summary is None or not self.terminal_summary.strip():
                raise ValueError("finished states must contain a terminal summary")
            if self.failure_reason is not None:
                raise ValueError("finished states must not contain a failure reason")
        elif self.phase is AgentPhase.FAILED:
            if self.failure_reason is None or not self.failure_reason.strip():
                raise ValueError("failed states must contain a failure reason")
            if self.terminal_summary is not None:
                raise ValueError("failed states must not contain a terminal summary")
        return self

    @classmethod
    def initial(cls, task_id: str) -> Self:
        """Create the initial state for a task investigation."""
        return cls(
            task_id=task_id,
            phase=AgentPhase.UNDERSTAND,
            issue_understanding=None,
            hypotheses=(),
            observations=(),
            step_count=0,
            terminal_summary=None,
            failure_reason=None,
        )
