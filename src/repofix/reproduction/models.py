"""Strict evaluator-only reproduction expectations, evidence, and verdicts."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Self

from pydantic import Field, field_validator, model_validator

from repofix.tasks.spec import AgentTaskSpec, StrictFrozenModel, validate_command_name

if TYPE_CHECKING:
    from repofix.execution import ApprovedCommandExecutionResult


MAX_REPRODUCTION_FRAGMENT_LENGTH = 4_096
_FRAGMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ReproductionOutputStream(StrEnum):
    """Command output stream searched for a literal fragment."""

    STDOUT = "stdout"
    STDERR = "stderr"
    COMBINED = "combined"


class ReproductionOutputFragment(StrictFrozenModel):
    """One evaluator-controlled literal output signature."""

    fragment_id: str
    stream: ReproductionOutputStream
    text: str = Field(max_length=MAX_REPRODUCTION_FRAGMENT_LENGTH)

    @field_validator("stream", mode="before")
    @classmethod
    def normalize_stream(cls, value: object) -> object:
        if isinstance(value, str):
            return ReproductionOutputStream(value)
        return value

    @field_validator("fragment_id")
    @classmethod
    def validate_fragment_id(cls, value: str) -> str:
        if not _FRAGMENT_ID_PATTERN.fullmatch(value):
            raise ValueError("fragment ID must be a stable lowercase identifier")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("fragment text must not be empty or whitespace")
        if "\0" in value:
            raise ValueError("fragment text must not contain NUL bytes")
        return value


class ReproductionExpectation(StrictFrozenModel):
    """Trusted evaluator rules for recognizing one reported failure."""

    command_id: str
    expected_exit_codes: tuple[int, ...]
    required_fragments: tuple[ReproductionOutputFragment, ...]
    forbidden_fragments: tuple[ReproductionOutputFragment, ...] = ()

    @field_validator(
        "expected_exit_codes",
        "required_fragments",
        "forbidden_fragments",
        mode="before",
    )
    @classmethod
    def normalize_sequence(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("reproduction sequences must be lists or tuples")
        return tuple(value)

    @field_validator("command_id")
    @classmethod
    def validate_command_id(cls, value: str) -> str:
        return validate_command_name(value)

    @field_validator("expected_exit_codes")
    @classmethod
    def validate_exit_codes(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("expected exit codes must not be empty")
        if any(value == 0 for value in values):
            raise ValueError("expected reproduction exit codes must be nonzero")
        if len(values) != len(set(values)):
            raise ValueError("expected exit codes must be unique")
        return values

    @model_validator(mode="after")
    def validate_fragments(self) -> Self:
        if not self.required_fragments:
            raise ValueError("at least one required output fragment is required")
        fragment_ids = [
            fragment.fragment_id
            for fragment in (*self.required_fragments, *self.forbidden_fragments)
        ]
        if len(fragment_ids) != len(set(fragment_ids)):
            raise ValueError("fragment IDs must be unique")
        return self


class ReproductionTaskBundle(StrictFrozenModel):
    """Agent task plus evaluator-only reproduction expectations."""

    task: AgentTaskSpec
    reproduction: ReproductionExpectation

    @model_validator(mode="after")
    def validate_command_reference(self) -> Self:
        if self.reproduction.command_id not in self.task.approved_commands:
            raise ValueError("reproduction command ID is not an approved task command")
        return self

    def agent_view(self) -> AgentTaskSpec:
        """Return only the existing agent-visible task model."""
        return self.task


class ReproductionTerminationReason(StrEnum):
    """Execution termination reason copied into reproduction evidence."""

    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    OUTPUT_LIMIT = "output_limit"


class ReproductionEvidence(StrictFrozenModel):
    """Bounded command evidence without a reproduction classification."""

    command_id: str
    argv: tuple[str, ...]
    termination_reason: ReproductionTerminationReason
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_bytes: int = Field(ge=0)
    stderr_bytes: int = Field(ge=0)
    had_decode_errors: bool

    @field_validator("command_id")
    @classmethod
    def validate_command_id(cls, value: str) -> str:
        return validate_command_name(value)

    @model_validator(mode="after")
    def validate_execution_shape(self) -> Self:
        if self.termination_reason is ReproductionTerminationReason.COMPLETED:
            if self.exit_code is None:
                raise ValueError("completed evidence requires an exit code")
        elif self.exit_code is not None:
            raise ValueError("bounded termination evidence requires exit_code=None")
        return self

    @classmethod
    def from_execution_result(
        cls, result: ApprovedCommandExecutionResult
    ) -> ReproductionEvidence:
        """Copy deterministic public evidence from one execution result."""
        from repofix.execution import ApprovedCommandExecutionResult

        if not isinstance(result, ApprovedCommandExecutionResult):
            raise TypeError("result must be an ApprovedCommandExecutionResult")
        return cls(
            command_id=result.command_id,
            argv=result.argv,
            termination_reason=ReproductionTerminationReason(
                result.termination_reason.value
            ),
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            stdout_bytes=result.stdout_bytes,
            stderr_bytes=result.stderr_bytes,
            had_decode_errors=result.had_decode_errors,
        )


class ReproductionStatus(StrEnum):
    """Deterministic reproduction classification."""

    REPRODUCED = "reproduced"
    NOT_REPRODUCED = "not_reproduced"
    INCONCLUSIVE = "inconclusive"


class ReproductionVerdict(StrictFrozenModel):
    """Concise reproduction verdict containing fragment IDs, never output text."""

    status: ReproductionStatus
    command_id: str
    exit_code: int | None
    reasons: tuple[str, ...]
    matched_required_fragment_ids: tuple[str, ...]
    missing_required_fragment_ids: tuple[str, ...]
    forbidden_fragment_ids_found: tuple[str, ...]

    @field_validator("command_id")
    @classmethod
    def validate_command_id(cls, value: str) -> str:
        return validate_command_name(value)

    @field_validator(
        "matched_required_fragment_ids",
        "missing_required_fragment_ids",
        "forbidden_fragment_ids_found",
    )
    @classmethod
    def validate_ordered_fragment_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _FRAGMENT_ID_PATTERN.fullmatch(value) for value in values):
            raise ValueError("verdict fragment IDs must be stable lowercase identifiers")
        if values != tuple(sorted(values)):
            raise ValueError("verdict fragment IDs must use deterministic ordering")
        return values

    @model_validator(mode="after")
    def validate_verdict(self) -> Self:
        if not self.reasons or any(not reason.strip() for reason in self.reasons):
            raise ValueError("verdict reasons must be nonempty")
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("verdict reasons must be unique")
        identifier_groups = (
            self.matched_required_fragment_ids,
            self.missing_required_fragment_ids,
            self.forbidden_fragment_ids_found,
        )
        if any(len(values) != len(set(values)) for values in identifier_groups):
            raise ValueError("verdict fragment IDs must be unique")
        if set(self.matched_required_fragment_ids) & set(
            self.missing_required_fragment_ids
        ):
            raise ValueError("required fragment IDs cannot be both matched and missing")
        if self.status is ReproductionStatus.REPRODUCED:
            if self.exit_code in {None, 0}:
                raise ValueError("reproduced verdicts require a nonzero exit code")
            if not self.matched_required_fragment_ids:
                raise ValueError("reproduced verdicts require matched output fragments")
            if self.missing_required_fragment_ids or self.forbidden_fragment_ids_found:
                raise ValueError("reproduced verdicts require all fragment checks to pass")
        if self.status is ReproductionStatus.NOT_REPRODUCED and self.exit_code != 0:
            raise ValueError("not-reproduced verdicts require exit code zero")
        return self
