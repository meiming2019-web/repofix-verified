"""Provider-independent records for one externally orchestrated repair run."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from repofix.patching.models import PatchApplicationStatus
from repofix.tasks.spec import StrictFrozenModel, validate_relative_source_path


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STATUS_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_ACTION_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?:^|[\s\"'(=+\-])[a-z]:[\\/]"
)
_ABSOLUTE_POSIX_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'(=+\-])/(?!/)[A-Za-z0-9._~@%+-]+"
    r"(?:/[A-Za-z0-9._~@%+-]+)*(?=$|[\s\"'),;:\]])"
)
_SECRET_LIKE_PATTERN = re.compile(
    r"(?i)(?:authorization\s*:|bearer\s+[A-Za-z0-9._-]+|"
    r"(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=])"
)
_PUBLIC_FORBIDDEN_MARKERS = (
    "OPENAI_API_KEY",
    "GoldPatchSpec",
    "gold_patch",
    "hidden_tests",
    "hidden_test_source",
    "repofix" + "-pilots",
)
_REPOSITORY_INSPECTION_ACTIONS = frozenset(
    {"list_files", "search_code", "read_file"}
)


class RunOutcome(StrEnum):
    """Terminal classification owned by the experiment recorder."""

    EVALUATOR_PASSED = "evaluator_passed"
    STOPPED_AT_REPRODUCTION = "stopped_at_reproduction"
    STOPPED_AT_PROPOSAL = "stopped_at_proposal"
    STOPPED_AT_REGRESSION_BASELINE = "stopped_at_regression_baseline"
    STOPPED_AT_APPLICATION = "stopped_at_application"
    STOPPED_AT_POST_PATCH_REPRODUCTION = "stopped_at_post_patch_reproduction"
    STOPPED_AT_REGRESSION = "stopped_at_regression"
    STOPPED_AT_HIDDEN = "stopped_at_hidden"
    STOPPED_AT_POLICY = "stopped_at_policy"
    MODEL_EXECUTION_BLOCKED = "model_execution_blocked"
    SYSTEM_ERROR = "system_error"


class RunStage(StrEnum):
    """Authoritative stages that an external driver may record."""

    REPRODUCTION = "reproduction"
    PROPOSAL = "proposal"
    REGRESSION_BASELINE = "regression_baseline"
    APPLICATION = "application"
    POST_PATCH_REPRODUCTION = "post_patch_reproduction"
    REGRESSION = "regression"
    HIDDEN = "hidden"
    POLICY = "policy"
    FINAL_EVALUATION = "final_evaluation"


_STAGE_ORDER = tuple(RunStage)
_STOP_STAGE_BY_OUTCOME = {
    RunOutcome.STOPPED_AT_REPRODUCTION: RunStage.REPRODUCTION,
    RunOutcome.STOPPED_AT_PROPOSAL: RunStage.PROPOSAL,
    RunOutcome.STOPPED_AT_REGRESSION_BASELINE: RunStage.REGRESSION_BASELINE,
    RunOutcome.STOPPED_AT_APPLICATION: RunStage.APPLICATION,
    RunOutcome.STOPPED_AT_POST_PATCH_REPRODUCTION: (
        RunStage.POST_PATCH_REPRODUCTION
    ),
    RunOutcome.STOPPED_AT_REGRESSION: RunStage.REGRESSION,
    RunOutcome.STOPPED_AT_HIDDEN: RunStage.HIDDEN,
    RunOutcome.STOPPED_AT_POLICY: RunStage.POLICY,
}
_SUCCESS_STATUS_BY_STAGE = {
    RunStage.REPRODUCTION: "reproduced",
    RunStage.PROPOSAL: "structurally_validated_unapplied",
    RunStage.REGRESSION_BASELINE: "passed",
    RunStage.APPLICATION: PatchApplicationStatus.APPLIED.value,
    RunStage.POST_PATCH_REPRODUCTION: "original_behavior_not_reproduced",
    RunStage.REGRESSION: "regression_command_passed",
    RunStage.HIDDEN: "hidden_command_passed",
    RunStage.POLICY: "policy_passed",
    RunStage.FINAL_EVALUATION: "evaluator_passed",
}


class ModelExecutionBlockCategory(StrEnum):
    """Sanitized reason that model execution could not proceed."""

    CREDENTIALS_UNAVAILABLE = "credentials_unavailable"
    CONFIGURATION_INVALID = "configuration_invalid"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    UNSUPPORTED_CONFIGURATION = "unsupported_configuration"


class SystemErrorCategory(StrEnum):
    """Non-evaluator category for an unexpected operator or runtime failure."""

    FILESYSTEM = "filesystem"
    INFRASTRUCTURE = "infrastructure"
    PROGRAMMER = "programmer"
    RUNTIME = "runtime"
    UNKNOWN = "unknown"


class RunTiming(StrictFrozenModel):
    """Timezone-aware wall-clock interval plus independently measured duration."""

    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)

    @field_validator("started_at", "finished_at", mode="before")
    @classmethod
    def parse_json_timestamp(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("run timestamp must use ISO 8601") from error
        return value

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run timestamps must be timezone-aware")
        return value

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("run duration must be finite")
        return value

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("run finish timestamp must not precede its start")
        return self


class RunIdentity(StrictFrozenModel):
    """Path-independent identity and declared configuration for one run."""

    run_id: str
    task_id: str
    task_fingerprint: str
    repository_url: str
    pre_fix_commit: str
    repofix_commit: str
    model_provider: str
    model_identifier: str
    agent_workflow: str
    timing: RunTiming | None = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not _RUN_ID_PATTERN.fullmatch(value):
            raise ValueError("run ID must be a stable path-free identifier")
        return value

    @field_validator("task_id", "model_provider", "model_identifier", "agent_workflow")
    @classmethod
    def validate_text_identity(cls, value: str) -> str:
        if not value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("run identity text must be nonempty and control-free")
        return value

    @field_validator("task_fingerprint")
    @classmethod
    def validate_task_fingerprint(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("task fingerprint must be lowercase SHA-256")
        return value

    @field_validator("pre_fix_commit", "repofix_commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if not _GIT_SHA_PATTERN.fullmatch(value):
            raise ValueError("run commits must be full lowercase 40-character Git SHAs")
        return value

    @field_validator("repository_url")
    @classmethod
    def validate_repository_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
        except ValueError as error:
            raise ValueError("run repository URL is malformed") from error
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            raise ValueError("run repository URL must be an absolute HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("run repository URL must not contain credentials")
        return value


class ModelCallUsage(StrictFrozenModel):
    """Optional provider-reported token counters for one model request."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)


def _sum_optional(values: tuple[int | None, ...]) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


class ModelUsage(StrictFrozenModel):
    """Provider-independent aggregate model usage without request identifiers."""

    model_call_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    configured_max_retries: int | None = Field(default=None, ge=0)
    provider_retry_count: int | None = Field(default=None, ge=0)
    structured_output_rejection_count: int | None = Field(default=None, ge=0)

    @classmethod
    def aggregate(
        cls,
        calls: tuple[ModelCallUsage, ...],
        *,
        configured_max_retries: int | None = None,
        provider_retry_count: int | None = None,
        structured_output_rejection_count: int | None = None,
    ) -> ModelUsage:
        """Aggregate only counters reported by every call."""
        return cls(
            model_call_count=len(calls),
            input_tokens=_sum_optional(tuple(call.input_tokens for call in calls)),
            output_tokens=_sum_optional(tuple(call.output_tokens for call in calls)),
            reasoning_tokens=_sum_optional(
                tuple(call.reasoning_tokens for call in calls)
            ),
            total_tokens=_sum_optional(tuple(call.total_tokens for call in calls)),
            cached_tokens=_sum_optional(tuple(call.cached_tokens for call in calls)),
            cache_write_tokens=_sum_optional(
                tuple(call.cache_write_tokens for call in calls)
            ),
            configured_max_retries=configured_max_retries,
            provider_retry_count=provider_retry_count,
            structured_output_rejection_count=(
                structured_output_rejection_count
            ),
        )


class AgentActionRecord(StrictFrozenModel):
    """One ordered action kind; sensitive action payloads remain private evidence."""

    index: int = Field(ge=1)
    kind: str

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        if not _ACTION_KIND_PATTERN.fullmatch(value):
            raise ValueError("agent action kind must be a stable lowercase identifier")
        return value


class ActionKindCount(StrictFrozenModel):
    kind: str
    count: int = Field(ge=1)

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        if not _ACTION_KIND_PATTERN.fullmatch(value):
            raise ValueError("action count kind must be a stable lowercase identifier")
        return value


class ActionCounts(StrictFrozenModel):
    """Unambiguous aggregate counts derived from ordered agent actions."""

    repository_inspection_calls: int = Field(ge=0)
    approved_command_requests: int = Field(ge=0)
    total_agent_actions: int = Field(ge=0)
    by_kind: tuple[ActionKindCount, ...]

    @field_validator("by_kind", mode="before")
    @classmethod
    def normalize_by_kind(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("action kind counts must be a list or tuple")
        return tuple(value)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        kinds = tuple(item.kind for item in self.by_kind)
        if kinds != tuple(sorted(kinds)) or len(kinds) != len(set(kinds)):
            raise ValueError("action kind counts must be sorted and unique")
        if sum(item.count for item in self.by_kind) != self.total_agent_actions:
            raise ValueError("action kind counts must sum to total agent actions")
        by_kind = {item.kind: item.count for item in self.by_kind}
        expected_repository = sum(
            by_kind.get(kind, 0) for kind in _REPOSITORY_INSPECTION_ACTIONS
        )
        if self.repository_inspection_calls != expected_repository:
            raise ValueError("repository inspection count does not match action kinds")
        if self.approved_command_requests != by_kind.get("run_approved_command", 0):
            raise ValueError("approved command count does not match action kinds")
        return self

    @classmethod
    def from_actions(cls, actions: tuple[AgentActionRecord, ...]) -> ActionCounts:
        counts: dict[str, int] = {}
        for action in actions:
            counts[action.kind] = counts.get(action.kind, 0) + 1
        return cls(
            repository_inspection_calls=sum(
                counts.get(kind, 0) for kind in _REPOSITORY_INSPECTION_ACTIONS
            ),
            approved_command_requests=counts.get("run_approved_command", 0),
            total_agent_actions=len(actions),
            by_kind=tuple(
                ActionKindCount(kind=kind, count=count)
                for kind, count in sorted(counts.items())
            ),
        )


class StageRecord(StrictFrozenModel):
    """Canonical status and optional identity/timing for one reached stage."""

    stage: RunStage
    status: str
    result_fingerprint: str | None = None
    finding_count: int | None = Field(default=None, ge=0)
    timing: RunTiming | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if not _STATUS_PATTERN.fullmatch(value):
            raise ValueError("stage status must be a canonical lowercase identifier")
        return value

    @field_validator("result_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("stage result fingerprint must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_finding_count(self) -> Self:
        if self.finding_count is not None and self.stage is not RunStage.POLICY:
            raise ValueError("only policy stages may record a finding count")
        return self


class PrivateCandidateProvenance(StrictFrozenModel):
    """Exact candidate material retained in the private experiment record."""

    proposal_digest: str
    target_paths: tuple[str, ...]
    edit_count: int = Field(ge=1)
    candidate_patch: str
    application_status: PatchApplicationStatus | None = None

    @field_validator("proposal_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("proposal digest must be lowercase SHA-256")
        return value

    @field_validator("target_paths", mode="before")
    @classmethod
    def normalize_paths(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("candidate target paths must be a list or tuple")
        return tuple(value)

    @field_validator("target_paths")
    @classmethod
    def validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("candidate target paths must not be empty")
        if value != tuple(sorted(value)) or len(value) != len(set(value)):
            raise ValueError("candidate target paths must be sorted and unique")
        for path in value:
            validate_relative_source_path(path, description="candidate target path")
        return value

    @field_validator("candidate_patch")
    @classmethod
    def validate_patch(cls, value: str) -> str:
        if not value or "\0" in value:
            raise ValueError("candidate patch must be nonempty and NUL-free")
        return value

class PublicCandidateProvenance(StrictFrozenModel):
    """Sanitized candidate identity with an optional publishable patch."""

    proposal_digest: str
    target_paths: tuple[str, ...]
    edit_count: int = Field(ge=1)
    candidate_patch: str | None
    patch_omitted_for_safety: bool
    application_status: PatchApplicationStatus | None = None

    @field_validator("proposal_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return PrivateCandidateProvenance.validate_digest(value)

    @field_validator("target_paths", mode="before")
    @classmethod
    def normalize_paths(cls, value: object) -> tuple[object, ...]:
        return PrivateCandidateProvenance.normalize_paths(value)

    @field_validator("target_paths")
    @classmethod
    def validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return PrivateCandidateProvenance.validate_paths(value)

    @model_validator(mode="after")
    def validate_patch_projection(self) -> Self:
        if self.patch_omitted_for_safety != (self.candidate_patch is None):
            raise ValueError("candidate patch omission marker must match patch presence")
        if self.candidate_patch is not None and not _is_public_patch_safe(
            self.candidate_patch
        ):
            raise ValueError("public candidate patch contains private material")
        return self


class ModelExecutionBlock(StrictFrozenModel):
    """Public-safe model configuration or provider blocking category."""

    category: ModelExecutionBlockCategory


class PrivateSystemError(StrictFrozenModel):
    """Internal error evidence; its message is never copied to a public record."""

    category: SystemErrorCategory
    error_type: str
    message: str

    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, value: str) -> str:
        if not _ERROR_TYPE_PATTERN.fullmatch(value):
            raise ValueError("system error type must be a stable class identifier")
        return value

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value or "\0" in value:
            raise ValueError("private system error message must be nonempty and NUL-free")
        return value


class PublicSystemError(StrictFrozenModel):
    """Sanitized system error category without arbitrary exception text."""

    category: SystemErrorCategory
    error_type: str

    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, value: str) -> str:
        return PrivateSystemError.validate_error_type(value)


class PrivateEvidenceRecord(StrictFrozenModel):
    """Opaque internal JSON evidence intentionally excluded from public records."""

    evidence_type: str
    payload_json: str

    @field_validator("evidence_type")
    @classmethod
    def validate_evidence_type(cls, value: str) -> str:
        if not _ACTION_KIND_PATTERN.fullmatch(value):
            raise ValueError("private evidence type must be a stable identifier")
        return value

    @field_validator("payload_json")
    @classmethod
    def validate_json(cls, value: str) -> str:
        try:
            json.loads(value)
        except (TypeError, ValueError) as error:
            raise ValueError("private evidence payload must be valid JSON") from error
        return value


def _is_public_patch_safe(value: str) -> bool:
    if "\0" in value:
        return False
    if _ABSOLUTE_POSIX_PATH_PATTERN.search(
        value
    ) or _WINDOWS_ABSOLUTE_PATH_PATTERN.search(value):
        return False
    if _SECRET_LIKE_PATTERN.search(value):
        return False
    return not any(marker in value for marker in _PUBLIC_FORBIDDEN_MARKERS)


def _normalize_model_tuple(value: object, *, description: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{description} must be a list or tuple")
    return tuple(value)


class _RunArtifactBase(StrictFrozenModel):
    schema_version: Literal[1] = 1
    identity: RunIdentity
    outcome: RunOutcome
    model_usage: ModelUsage
    actions: tuple[AgentActionRecord, ...]
    stages: tuple[StageRecord, ...]
    candidate: PrivateCandidateProvenance | PublicCandidateProvenance | None = None
    model_execution_block: ModelExecutionBlock | None = None

    @field_validator("actions", mode="before")
    @classmethod
    def normalize_actions(cls, value: object) -> tuple[object, ...]:
        return _normalize_model_tuple(value, description="run actions")

    @field_validator("stages", mode="before")
    @classmethod
    def normalize_stages(cls, value: object) -> tuple[object, ...]:
        return _normalize_model_tuple(value, description="run stages")

    @model_validator(mode="after")
    def validate_run_shape(self) -> Self:
        indices = tuple(action.index for action in self.actions)
        if indices != tuple(range(1, len(self.actions) + 1)):
            raise ValueError("run actions must use contiguous one-based ordering")

        stages = tuple(record.stage for record in self.stages)
        if stages != _STAGE_ORDER[: len(stages)]:
            raise ValueError("run stages must form a contiguous authoritative prefix")

        for record in self.stages[:-1]:
            if record.status != _SUCCESS_STATUS_BY_STAGE[record.stage]:
                raise ValueError("every completed prerequisite stage must have succeeded")

        if self.outcome is RunOutcome.EVALUATOR_PASSED:
            if stages != _STAGE_ORDER:
                raise ValueError("evaluator-passed runs require the complete stage chain")
            if self.stages[-1].status != _SUCCESS_STATUS_BY_STAGE[RunStage.FINAL_EVALUATION]:
                raise ValueError("evaluator-passed runs require the final success status")
            if self.candidate is None:
                raise ValueError("evaluator-passed runs require candidate provenance")
            if self.candidate.application_status is not PatchApplicationStatus.APPLIED:
                raise ValueError("evaluator-passed runs require an applied candidate")
        elif RunStage.FINAL_EVALUATION in stages:
            raise ValueError("only evaluator-passed runs may contain final evaluation")

        stop_stage = _STOP_STAGE_BY_OUTCOME.get(self.outcome)
        if stop_stage is not None:
            stop_index = _STAGE_ORDER.index(stop_stage)
            if stages != _STAGE_ORDER[: stop_index + 1]:
                raise ValueError("workflow-stop run must end at its declared stage")
            if self.stages[-1].status == _SUCCESS_STATUS_BY_STAGE[stop_stage]:
                raise ValueError("workflow-stop stage must not carry its success status")

        if RunStage.REGRESSION_BASELINE in stages and self.candidate is None:
            raise ValueError("regression baseline and later stages require a candidate")
        application = next(
            (record for record in self.stages if record.stage is RunStage.APPLICATION),
            None,
        )
        if application is not None and application.status == PatchApplicationStatus.APPLIED.value:
            if (
                self.candidate is None
                or self.candidate.application_status is not PatchApplicationStatus.APPLIED
            ):
                raise ValueError("a successful application stage requires an applied candidate")
        if (
            self.candidate is not None
            and self.candidate.application_status is PatchApplicationStatus.APPLIED
            and (application is None or application.status != PatchApplicationStatus.APPLIED.value)
        ):
            raise ValueError("an applied candidate requires a successful application stage")

        if (
            self.outcome is RunOutcome.MODEL_EXECUTION_BLOCKED
        ) != (self.model_execution_block is not None):
            raise ValueError("model block details must match model-execution-blocked outcome")
        return self

    def to_canonical_json(self) -> str:
        """Serialize deterministically for storage or hashing."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class PrivateRunArtifact(_RunArtifactBase):
    """Internal record that may retain raw evidence and exception text."""

    candidate: PrivateCandidateProvenance | None = None
    system_error: PrivateSystemError | None = None
    private_evidence: tuple[PrivateEvidenceRecord, ...] = ()

    @field_validator("private_evidence", mode="before")
    @classmethod
    def normalize_private_evidence(cls, value: object) -> tuple[object, ...]:
        return _normalize_model_tuple(value, description="private evidence")

    @model_validator(mode="after")
    def validate_system_error(self) -> Self:
        if (self.outcome is RunOutcome.SYSTEM_ERROR) != (self.system_error is not None):
            raise ValueError("private system error must match system-error outcome")
        return self

    def to_public(self) -> PublicRunArtifact:
        """Project private evidence into a structurally limited public record."""
        public_candidate: PublicCandidateProvenance | None = None
        if self.candidate is not None:
            safe_patch = (
                self.candidate.candidate_patch
                if _is_public_patch_safe(self.candidate.candidate_patch)
                else None
            )
            public_candidate = PublicCandidateProvenance(
                proposal_digest=self.candidate.proposal_digest,
                target_paths=self.candidate.target_paths,
                edit_count=self.candidate.edit_count,
                candidate_patch=safe_patch,
                patch_omitted_for_safety=safe_patch is None,
                application_status=self.candidate.application_status,
            )
        public_error = (
            PublicSystemError(
                category=self.system_error.category,
                error_type=self.system_error.error_type,
            )
            if self.system_error is not None
            else None
        )
        return PublicRunArtifact(
            identity=self.identity,
            outcome=self.outcome,
            model_usage=self.model_usage,
            actions=self.actions,
            action_counts=ActionCounts.from_actions(self.actions),
            stages=self.stages,
            candidate=public_candidate,
            model_execution_block=self.model_execution_block,
            system_error=public_error,
        )


class PublicRunArtifact(_RunArtifactBase):
    """Sanitized, deterministic record safe for repository storage."""

    candidate: PublicCandidateProvenance | None = None
    action_counts: ActionCounts
    system_error: PublicSystemError | None = None

    @model_validator(mode="after")
    def validate_public_record(self) -> Self:
        expected_counts = ActionCounts.from_actions(self.actions)
        if self.action_counts != expected_counts:
            raise ValueError("public action counts must be derived from recorded actions")
        if (self.outcome is RunOutcome.SYSTEM_ERROR) != (self.system_error is not None):
            raise ValueError("public system error must match system-error outcome")
        serialized = self.to_canonical_json()
        if any(marker in serialized for marker in _PUBLIC_FORBIDDEN_MARKERS):
            raise ValueError("public run record contains evaluator-private material")
        if _SECRET_LIKE_PATTERN.search(serialized):
            raise ValueError("public run record contains secret-like material")
        if _ABSOLUTE_POSIX_PATH_PATTERN.search(
            serialized
        ) or _WINDOWS_ABSOLUTE_PATH_PATTERN.search(serialized):
            raise ValueError("public run record contains an absolute host path")
        return self


def compute_public_run_artifact_fingerprint(artifact: PublicRunArtifact) -> str:
    """Hash one canonical public artifact without private evidence."""
    return hashlib.sha256(artifact.to_canonical_json().encode("utf-8")).hexdigest()
