"""Structured actions available to the read-only investigation agent."""

from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from repofix.agent.state import IssueUnderstanding, RepairHypothesis
from repofix.tasks.spec import validate_command_name


class _ActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class UnderstandIssueAction(_ActionModel):
    """Record a structured understanding of the issue."""

    kind: Literal["understand_issue"]
    understanding: IssueUnderstanding


class ListFilesAction(_ActionModel):
    """Request a read-only directory listing."""

    kind: Literal["list_files"]
    path: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must not be empty")
        return value


class SearchCodeAction(_ActionModel):
    """Request a read-only code search."""

    kind: Literal["search_code"]
    query: str
    file_glob: str | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


class ReadFileAction(_ActionModel):
    """Request a bounded range from a source file."""

    kind: Literal["read_file"]
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must not be empty")
        return value

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("end line must not precede start line")
        return self


class RecordHypothesisAction(_ActionModel):
    """Record an evidence-linked repair hypothesis."""

    kind: Literal["record_hypothesis"]
    hypothesis: RepairHypothesis


class RunApprovedCommandAction(_ActionModel):
    """Request one trusted TaskSpec command by its exact identifier."""

    kind: Literal["run_approved_command"] = "run_approved_command"
    command_id: str

    @field_validator("command_id")
    @classmethod
    def validate_command_id(cls, value: str) -> str:
        return validate_command_name(value)


class FinishInvestigationAction(_ActionModel):
    """End the investigation without claiming the repair is resolved."""

    kind: Literal["finish_investigation"]
    summary: str

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("investigation summary must not be empty")
        return value


InvestigationAgentAction: TypeAlias = Annotated[
    UnderstandIssueAction
    | ListFilesAction
    | SearchCodeAction
    | ReadFileAction
    | RecordHypothesisAction
    | FinishInvestigationAction,
    Field(discriminator="kind"),
]

ReproductionPreAttemptAgentAction: TypeAlias = Annotated[
    UnderstandIssueAction
    | ListFilesAction
    | SearchCodeAction
    | ReadFileAction
    | RecordHypothesisAction
    | RunApprovedCommandAction,
    Field(discriminator="kind"),
]

ReproductionPostAttemptAgentAction: TypeAlias = Annotated[
    UnderstandIssueAction
    | ListFilesAction
    | SearchCodeAction
    | ReadFileAction
    | RecordHypothesisAction,
    Field(discriminator="kind"),
]

AgentAction: TypeAlias = Annotated[
    UnderstandIssueAction
    | ListFilesAction
    | SearchCodeAction
    | ReadFileAction
    | RecordHypothesisAction
    | RunApprovedCommandAction
    | FinishInvestigationAction,
    Field(discriminator="kind"),
]
