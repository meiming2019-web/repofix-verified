"""Validated data models for RepoFix task specifications."""

import re
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


_COMMAND_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def _validate_command_mapping(
    commands: dict[str, "ApprovedCommand"],
) -> dict[str, "ApprovedCommand"]:
    if not commands:
        raise ValueError("commands must not be empty")
    for name in commands:
        if not _COMMAND_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid command name: {name!r}")
    return commands


class StrictFrozenModel(BaseModel):
    """Base configuration shared by task specification models."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ApprovedCommand(StrictFrozenModel):
    """A command approved as an explicit argument vector."""

    argv: tuple[str, ...]

    @field_validator("argv", mode="before")
    @classmethod
    def normalize_argv_container(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("argv must be a list or tuple")
        return tuple(value)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, argv: tuple[str, ...]) -> tuple[str, ...]:
        if not argv:
            raise ValueError("argv must not be empty")
        if not argv[0].strip():
            raise ValueError("the executable must not be empty or whitespace")
        if any("\0" in argument for argument in argv):
            raise ValueError("command arguments must not contain NUL bytes")
        return argv


class AgentTaskSpec(StrictFrozenModel):
    """Task information that may be exposed to the repair agent."""

    task_id: str
    repository_url: str
    pre_fix_commit: str
    issue_title: str
    issue_body: str
    approved_commands: dict[str, ApprovedCommand]
    allowed_source_paths: tuple[str, ...]
    timeout_seconds: int = Field(ge=1, le=3600)

    @field_validator("task_id", "issue_title")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty or whitespace")
        return value

    @field_validator("repository_url")
    @classmethod
    def validate_repository_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
        except ValueError as error:
            raise ValueError("repository URL is malformed") from error
        if parsed.scheme.lower() != "https":
            raise ValueError("repository URL must use HTTPS")
        if not host:
            raise ValueError("repository URL must contain a host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("repository URL must not contain credentials")
        return value

    @field_validator("pre_fix_commit")
    @classmethod
    def validate_pre_fix_commit(cls, value: str) -> str:
        if not _COMMIT_PATTERN.fullmatch(value):
            raise ValueError("pre-fix commit must be a full 40-character hexadecimal SHA")
        return value

    @field_validator("approved_commands")
    @classmethod
    def validate_approved_commands(
        cls, commands: dict[str, ApprovedCommand]
    ) -> dict[str, ApprovedCommand]:
        return _validate_command_mapping(commands)

    @field_validator("allowed_source_paths", mode="before")
    @classmethod
    def normalize_allowed_source_paths_container(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("allowed source paths must be a list or tuple")
        return tuple(value)

    @field_validator("allowed_source_paths")
    @classmethod
    def validate_allowed_source_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        if not paths:
            raise ValueError("allowed source paths must not be empty")
        for path in paths:
            if not path:
                raise ValueError("allowed source paths must not be empty")
            if "\0" in path:
                raise ValueError("allowed source paths must not contain NUL bytes")
            if "\\" in path:
                raise ValueError("allowed source paths must use POSIX separators")
            if path.startswith("/"):
                raise ValueError("allowed source paths must be repository-relative")
            components = path.split("/")
            if any(component in {"", ".", ".."} for component in components):
                raise ValueError("allowed source paths must not contain redundant components")
        return paths


class HiddenTestSpec(StrictFrozenModel):
    """Evaluator-only commands used to run hidden tests."""

    commands: dict[str, ApprovedCommand]

    @field_validator("commands")
    @classmethod
    def validate_commands(
        cls, commands: dict[str, ApprovedCommand]
    ) -> dict[str, ApprovedCommand]:
        return _validate_command_mapping(commands)


class GoldPatchSpec(StrictFrozenModel):
    """Evaluator-only reference patch data."""

    patch: str = Field(min_length=1)


class EvaluatorTaskBundle(StrictFrozenModel):
    """Complete evaluator data with an explicit agent-facing boundary."""

    task: AgentTaskSpec
    hidden_tests: HiddenTestSpec
    gold_patch: GoldPatchSpec

    def agent_view(self) -> AgentTaskSpec:
        """Return only the task information intended for the agent."""
        return self.task
