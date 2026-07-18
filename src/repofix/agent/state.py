"""Strict public state models for read-only agent workflows."""

from enum import Enum, StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from repofix.execution import CommandTerminationReason
from repofix.reproduction.models import ReproductionStatus
from repofix.tasks.spec import validate_command_name


REPRODUCED_TERMINAL_SUMMARY = (
    "The reported behavior was reproduced. No patch was generated or verified."
)


class AgentWorkflow(StrEnum):
    """Explicit mode controlling the actions available to an agent."""

    INVESTIGATION = "investigation"
    REPRODUCTION = "reproduction"


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


class AgentReproductionObservation(_StrictFrozenModel):
    """Sanitized public evidence and status from one approved command."""

    command_id: str
    termination_reason: CommandTerminationReason
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_bytes: int = Field(ge=0)
    stderr_bytes: int = Field(ge=0)
    had_decode_errors: bool
    status: ReproductionStatus

    @field_validator("command_id")
    @classmethod
    def validate_command_id(cls, value: str) -> str:
        return validate_command_name(value)

    @model_validator(mode="after")
    def validate_execution_shape(self) -> Self:
        if self.termination_reason is CommandTerminationReason.COMPLETED:
            if self.exit_code is None:
                raise ValueError("completed observations require an exit code")
        elif self.exit_code is not None:
            raise ValueError("bounded termination observations require exit_code=None")
        if self.status is ReproductionStatus.REPRODUCED and (
            self.termination_reason is not CommandTerminationReason.COMPLETED
            or self.exit_code == 0
        ):
            raise ValueError("reproduced observations require completed nonzero execution")
        if self.status is ReproductionStatus.NOT_REPRODUCED and (
            self.termination_reason is not CommandTerminationReason.COMPLETED
            or self.exit_code != 0
        ):
            raise ValueError("not-reproduced observations require completed zero exit")
        if (
            self.termination_reason
            in {CommandTerminationReason.TIMED_OUT, CommandTerminationReason.OUTPUT_LIMIT}
            and self.status is not ReproductionStatus.INCONCLUSIVE
        ):
            raise ValueError("bounded termination observations must be inconclusive")
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
    workflow: AgentWorkflow = AgentWorkflow.INVESTIGATION
    reproduction_command_id: str | None = None
    reproduction_observations: tuple[AgentReproductionObservation, ...] = ()

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
        if self.workflow is AgentWorkflow.INVESTIGATION:
            if (
                self.reproduction_command_id is not None
                or self.reproduction_observations
            ):
                raise ValueError("investigation states cannot contain reproduction results")
        elif self.reproduction_command_id is None:
            raise ValueError("reproduction states require a configured command ID")
        else:
            validate_command_name(self.reproduction_command_id)
            if len(self.reproduction_observations) > 1:
                raise ValueError("reproduction states permit at most one observation")
            if any(
                observation.command_id != self.reproduction_command_id
                for observation in self.reproduction_observations
            ):
                raise ValueError("reproduction observations must use the configured command ID")
        if self.phase is not AgentPhase.FINISHED and any(
            observation.status is ReproductionStatus.REPRODUCED
            for observation in self.reproduction_observations
        ):
            raise ValueError("reproduced evidence requires a finished state")
        if self.workflow is AgentWorkflow.REPRODUCTION and self.phase is AgentPhase.FINISHED:
            if (
                len(self.reproduction_observations) != 1
                or self.reproduction_observations[-1].status
                is not ReproductionStatus.REPRODUCED
            ):
                raise ValueError("finished reproduction states require exactly one reproduced observation")
            if self.terminal_summary != REPRODUCED_TERMINAL_SUMMARY:
                raise ValueError("finished reproduction states require the canonical summary")
        return self

    @classmethod
    def initial(
        cls,
        task_id: str,
        workflow: AgentWorkflow = AgentWorkflow.INVESTIGATION,
        reproduction_command_id: str | None = None,
    ) -> Self:
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
            workflow=workflow,
            reproduction_command_id=reproduction_command_id,
            reproduction_observations=(),
        )
