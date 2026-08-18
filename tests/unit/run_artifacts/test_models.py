"""Tests for private and sanitized public experiment records."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from repofix.patching import PatchApplicationStatus
from repofix.run_artifacts import (
    ActionCounts,
    AgentActionRecord,
    ModelCallUsage,
    ModelExecutionBlock,
    ModelExecutionBlockCategory,
    ModelUsage,
    PrivateCandidateProvenance,
    PrivateEvidenceRecord,
    PrivateRunArtifact,
    PrivateSystemError,
    PublicRunArtifact,
    RunIdentity,
    RunOutcome,
    RunStage,
    RunTiming,
    StageRecord,
    SystemErrorCategory,
    compute_public_run_artifact_fingerprint,
)


def identity(**updates: object) -> RunIdentity:
    values: dict[str, object] = {
        "run_id": "fixture-run-001",
        "task_id": "fixture-task",
        "task_fingerprint": "a" * 64,
        "repository_url": "https://example.com/owner/repository.git",
        "pre_fix_commit": "b" * 40,
        "repofix_commit": "c" * 40,
        "model_provider": "provider",
        "model_identifier": "model-1",
        "agent_workflow": "reproduction",
        "timing": RunTiming(
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            finished_at=datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC),
            duration_seconds=4.75,
        ),
    }
    values.update(updates)
    return RunIdentity.model_validate(values)


def actions() -> tuple[AgentActionRecord, ...]:
    kinds = (
        "understand_issue",
        "list_files",
        "search_code",
        "read_file",
        "record_hypothesis",
        "run_approved_command",
        "future_action",
    )
    return tuple(
        AgentActionRecord(index=index, kind=kind)
        for index, kind in enumerate(kinds, start=1)
    )


def usage() -> ModelUsage:
    return ModelUsage(
        model_call_count=3,
        input_tokens=30,
        output_tokens=12,
        reasoning_tokens=4,
        total_tokens=42,
        cached_tokens=5,
        cache_write_tokens=20,
        configured_max_retries=0,
        provider_retry_count=0,
        structured_output_rejection_count=0,
    )


def successful_stages() -> tuple[StageRecord, ...]:
    statuses = (
        (RunStage.REPRODUCTION, "reproduced"),
        (RunStage.PROPOSAL, "structurally_validated_unapplied"),
        (RunStage.REGRESSION_BASELINE, "passed"),
        (RunStage.APPLICATION, "applied"),
        (RunStage.POST_PATCH_REPRODUCTION, "original_behavior_not_reproduced"),
        (RunStage.REGRESSION, "regression_command_passed"),
        (RunStage.HIDDEN, "hidden_command_passed"),
        (RunStage.POLICY, "policy_passed"),
        (RunStage.FINAL_EVALUATION, "evaluator_passed"),
    )
    return tuple(
        StageRecord(
            stage=stage,
            status=status,
            result_fingerprint=str(index) * 64,
            finding_count=0 if stage is RunStage.POLICY else None,
            timing=RunTiming(
                started_at=datetime(2026, 1, 1, 0, 0, index, tzinfo=UTC),
                finished_at=datetime(
                    2026, 1, 1, 0, 0, index, tzinfo=UTC
                )
                + timedelta(milliseconds=250),
                duration_seconds=0.24,
            ),
        )
        for index, (stage, status) in enumerate(statuses, start=1)
    )


def candidate(patch: str | None = None) -> PrivateCandidateProvenance:
    return PrivateCandidateProvenance(
        proposal_digest="d" * 64,
        target_paths=("src/module.py",),
        edit_count=1,
        candidate_patch=patch
        or "diff --git a/src/module.py b/src/module.py\n-old\n+new\n",
        application_status=PatchApplicationStatus.APPLIED,
    )


def successful_private(**updates: object) -> PrivateRunArtifact:
    values: dict[str, object] = {
        "identity": identity(),
        "outcome": RunOutcome.EVALUATOR_PASSED,
        "model_usage": usage(),
        "actions": actions(),
        "stages": successful_stages(),
        "candidate": candidate(),
        "private_evidence": (),
    }
    values.update(updates)
    return PrivateRunArtifact.model_validate(values)


def test_outcome_taxonomy_separates_gates_model_blocks_and_system_errors() -> None:
    assert tuple(RunOutcome) == (
        RunOutcome.EVALUATOR_PASSED,
        RunOutcome.STOPPED_AT_REPRODUCTION,
        RunOutcome.STOPPED_AT_PROPOSAL,
        RunOutcome.STOPPED_AT_REGRESSION_BASELINE,
        RunOutcome.STOPPED_AT_APPLICATION,
        RunOutcome.STOPPED_AT_POST_PATCH_REPRODUCTION,
        RunOutcome.STOPPED_AT_REGRESSION,
        RunOutcome.STOPPED_AT_HIDDEN,
        RunOutcome.STOPPED_AT_POLICY,
        RunOutcome.MODEL_EXECUTION_BLOCKED,
        RunOutcome.SYSTEM_ERROR,
    )


def test_successful_models_are_strict_frozen_and_deterministic() -> None:
    private = successful_private()
    public = private.to_public()

    assert public.outcome is RunOutcome.EVALUATOR_PASSED
    assert public.to_canonical_json() == public.to_canonical_json()
    assert compute_public_run_artifact_fingerprint(public) == (
        compute_public_run_artifact_fingerprint(
            PublicRunArtifact.model_validate_json(public.to_canonical_json())
        )
    )
    with pytest.raises(ValidationError):
        public.outcome = RunOutcome.SYSTEM_ERROR  # type: ignore[misc]
    with pytest.raises(ValidationError):
        PrivateRunArtifact.model_validate(
            {**private.model_dump(), "unknown": "forbidden"}
        )


@pytest.mark.parametrize(
    "commit",
    ["c" * 39, "c" * 41, "C" * 40, "not-a-commit"],
)
def test_repofix_commit_requires_full_canonical_git_sha(commit: str) -> None:
    with pytest.raises(ValidationError, match="full lowercase 40-character"):
        identity(repofix_commit=commit)


def test_early_authoritative_stop_requires_stage_and_omits_later_stages() -> None:
    stopped = PrivateRunArtifact(
        identity=identity(),
        outcome=RunOutcome.STOPPED_AT_PROPOSAL,
        model_usage=usage(),
        actions=actions(),
        stages=(
            StageRecord(stage=RunStage.REPRODUCTION, status="reproduced"),
            StageRecord(stage=RunStage.PROPOSAL, status="invalid"),
        ),
    )

    assert stopped.to_public().outcome is RunOutcome.STOPPED_AT_PROPOSAL
    assert [record.stage for record in stopped.stages] == [
        RunStage.REPRODUCTION,
        RunStage.PROPOSAL,
    ]
    with pytest.raises(ValidationError, match="contiguous authoritative prefix"):
        PrivateRunArtifact.model_validate(
            {
                **stopped.model_dump(),
                "stages": (
                    *stopped.stages,
                    StageRecord(stage=RunStage.APPLICATION, status="applied"),
                ),
            }
        )


def test_model_execution_block_is_not_a_workflow_stop() -> None:
    artifact = PrivateRunArtifact(
        identity=identity(),
        outcome=RunOutcome.MODEL_EXECUTION_BLOCKED,
        model_usage=ModelUsage(model_call_count=0),
        actions=(),
        stages=(),
        model_execution_block=ModelExecutionBlock(
            category=ModelExecutionBlockCategory.CREDENTIALS_UNAVAILABLE
        ),
    )

    public = artifact.to_public()
    assert public.model_execution_block is not None
    assert public.model_execution_block.category is (
        ModelExecutionBlockCategory.CREDENTIALS_UNAVAILABLE
    )
    assert public.system_error is None


def test_evaluator_passed_rejects_only_final_stage() -> None:
    with pytest.raises(ValidationError, match="contiguous authoritative prefix"):
        successful_private(
            stages=(
                StageRecord(
                    stage=RunStage.FINAL_EVALUATION,
                    status="evaluator_passed",
                ),
            )
        )


def test_evaluator_passed_requires_candidate_provenance() -> None:
    with pytest.raises(ValidationError, match="require candidate provenance"):
        successful_private(candidate=None)


def test_stopped_at_hidden_rejects_lone_hidden_stage() -> None:
    with pytest.raises(ValidationError, match="contiguous authoritative prefix"):
        PrivateRunArtifact(
            identity=identity(),
            outcome=RunOutcome.STOPPED_AT_HIDDEN,
            model_usage=usage(),
            actions=actions(),
            stages=(StageRecord(stage=RunStage.HIDDEN, status="failed"),),
        )


def test_stage_sequence_rejects_a_gap() -> None:
    with pytest.raises(ValidationError, match="contiguous authoritative prefix"):
        PrivateRunArtifact(
            identity=identity(),
            outcome=RunOutcome.SYSTEM_ERROR,
            model_usage=usage(),
            actions=actions(),
            stages=(
                StageRecord(stage=RunStage.REPRODUCTION, status="reproduced"),
                StageRecord(stage=RunStage.HIDDEN, status="failed"),
            ),
            system_error=PrivateSystemError(
                category=SystemErrorCategory.RUNTIME,
                error_type="RuntimeError",
                message="operator failed",
            ),
        )


@pytest.mark.parametrize(
    ("outcome", "terminal_stage", "terminal_status"),
    [
        (
            RunOutcome.STOPPED_AT_HIDDEN,
            RunStage.HIDDEN,
            "hidden_command_passed",
        ),
        (
            RunOutcome.STOPPED_AT_REGRESSION,
            RunStage.REGRESSION,
            "regression_command_passed",
        ),
    ],
)
def test_workflow_stop_rejects_terminal_stage_success(
    outcome: RunOutcome,
    terminal_stage: RunStage,
    terminal_status: str,
) -> None:
    terminal_index = tuple(RunStage).index(terminal_stage)
    stages = tuple(
        StageRecord(
            stage=record.stage,
            status=(
                terminal_status
                if record.stage is terminal_stage
                else record.status
            ),
        )
        for record in successful_stages()[: terminal_index + 1]
    )

    with pytest.raises(ValidationError, match="must not carry its success status"):
        PrivateRunArtifact(
            identity=identity(),
            outcome=outcome,
            model_usage=usage(),
            actions=actions(),
            stages=stages,
            candidate=candidate(),
        )


def test_later_than_proposal_stage_requires_candidate() -> None:
    with pytest.raises(ValidationError, match="later stages require a candidate"):
        PrivateRunArtifact(
            identity=identity(),
            outcome=RunOutcome.STOPPED_AT_REGRESSION_BASELINE,
            model_usage=usage(),
            actions=actions(),
            stages=(
                StageRecord(stage=RunStage.REPRODUCTION, status="reproduced"),
                StageRecord(
                    stage=RunStage.PROPOSAL,
                    status="structurally_validated_unapplied",
                ),
                StageRecord(stage=RunStage.REGRESSION_BASELINE, status="failed"),
            ),
        )


def test_system_error_is_sanitized_and_distinct_from_workflow_stop() -> None:
    private_path = "/" + "Users" + "/someone/private/workspace"
    artifact = PrivateRunArtifact(
        identity=identity(),
        outcome=RunOutcome.SYSTEM_ERROR,
        model_usage=ModelUsage(model_call_count=0),
        actions=(),
        stages=(),
        system_error=PrivateSystemError(
            category=SystemErrorCategory.FILESYSTEM,
            error_type="FileNotFoundError",
            message=f"missing operator file at {private_path}",
        ),
        private_evidence=(
            PrivateEvidenceRecord(
                evidence_type="operator_error",
                payload_json='{"OPENAI_API_KEY":"private-value"}',
            ),
        ),
    )

    public = artifact.to_public()
    serialized = public.to_canonical_json()
    assert public.outcome is RunOutcome.SYSTEM_ERROR
    assert public.system_error is not None
    assert public.system_error.error_type == "FileNotFoundError"
    assert "missing operator file" not in serialized
    assert private_path not in serialized
    assert "private-value" not in serialized
    assert "private_evidence" not in serialized


def test_model_usage_aggregation_preserves_optional_provider_fields() -> None:
    complete = ModelCallUsage(
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=2,
        total_tokens=15,
        cached_tokens=3,
        cache_write_tokens=7,
    )
    partial = ModelCallUsage(
        input_tokens=20,
        output_tokens=6,
        total_tokens=26,
    )

    aggregated = ModelUsage.aggregate(
        (complete, complete),
        configured_max_retries=0,
        provider_retry_count=0,
        structured_output_rejection_count=1,
    )
    incomplete = ModelUsage.aggregate((complete, partial))

    assert aggregated.model_dump() == {
        "model_call_count": 2,
        "input_tokens": 20,
        "output_tokens": 10,
        "reasoning_tokens": 4,
        "total_tokens": 30,
        "cached_tokens": 6,
        "cache_write_tokens": 14,
        "configured_max_retries": 0,
        "provider_retry_count": 0,
        "structured_output_rejection_count": 1,
    }
    assert incomplete.input_tokens == 30
    assert incomplete.output_tokens == 11
    assert incomplete.reasoning_tokens is None
    assert incomplete.cached_tokens is None


def test_action_counts_separate_repository_commands_and_total_actions() -> None:
    counts = ActionCounts.from_actions(actions())

    assert counts.repository_inspection_calls == 3
    assert counts.approved_command_requests == 1
    assert counts.total_agent_actions == 7
    assert {item.kind: item.count for item in counts.by_kind}["future_action"] == 1


@pytest.mark.parametrize(
    "timing",
    [
        {
            "started_at": datetime(2026, 1, 1),
            "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
            "duration_seconds": 0.0,
        },
        {
            "started_at": datetime(2026, 1, 2, tzinfo=UTC),
            "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
            "duration_seconds": 0.0,
        },
        {
            "started_at": datetime(2026, 1, 1, tzinfo=UTC),
            "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
            "duration_seconds": float("inf"),
        },
    ],
)
def test_timing_rejects_naive_reversed_or_nonfinite_values(
    timing: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        RunTiming.model_validate(timing)


def test_candidate_provenance_is_preserved_when_public_safe() -> None:
    public = successful_private().to_public()

    assert public.candidate is not None
    assert public.candidate.proposal_digest == "d" * 64
    assert public.candidate.target_paths == ("src/module.py",)
    assert public.candidate.edit_count == 1
    assert public.candidate.application_status == "applied"
    assert public.candidate.candidate_patch is not None
    assert public.candidate.patch_omitted_for_safety is False


@pytest.mark.parametrize(
    "private_value",
    [
        "/tmp",
        "/private",
        "/Users",
        "/" + "Users" + "/someone/work/file.py",
        "/private" + "/tmp/evaluator/file.py",
        "/opt/toolchain/bin/python",
        "hidden" + "_tests/evaluator_contract.py",
        "Gold" + "PatchSpec",
        "OPENAI" + "_API_KEY=private-value",
        "Authorization" + ": Bearer private-value",
        "repofix" + "-pilots/private-workspace",
    ],
)
def test_public_projection_omits_unsafe_candidate_patch(private_value: str) -> None:
    private = successful_private(candidate=candidate(f"+{private_value}\n"))

    public = private.to_public()
    serialized = public.to_canonical_json()

    assert public.candidate is not None
    assert public.candidate.candidate_patch is None
    assert public.candidate.patch_omitted_for_safety is True
    assert private_value not in serialized


def test_public_artifact_permits_normal_https_repository_url() -> None:
    repository_url = "https://github.com/python-attrs/attrs.git"
    public = successful_private(
        identity=identity(repository_url=repository_url)
    ).to_public()

    assert public.identity.repository_url == repository_url
    assert repository_url in public.to_canonical_json()


def test_public_shape_has_no_raw_hidden_policy_gold_or_environment_fields() -> None:
    fields = set(PublicRunArtifact.model_fields)

    assert fields.isdisjoint(
        {
            "private_evidence",
            "environment",
            "response_ids",
            "hidden_command",
            "hidden_evidence",
            "policy_specification",
            "gold_patch",
            "exception_message",
            "workspace_root",
        }
    )
