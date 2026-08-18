"""Tests for the minimal terminal evaluation model."""

import pytest
from pydantic import ValidationError

from repofix.final_evaluation import (
    FINAL_EVALUATION_PASSED_SUMMARY,
    FinalEvaluationResult,
    FinalEvaluationStatus,
)


def values() -> dict[str, object]:
    return {
        "task_id": "task",
        "task_fingerprint": "a" * 64,
        "proposal_digest": "b" * 64,
        "policy_verification_fingerprint": "c" * 64,
        "status": FinalEvaluationStatus.EVALUATOR_PASSED,
        "verification_summary": FINAL_EVALUATION_PASSED_SUMMARY,
    }


def test_final_status_has_exactly_one_conservative_value() -> None:
    assert list(FinalEvaluationStatus) == [FinalEvaluationStatus.EVALUATOR_PASSED]
    assert FinalEvaluationStatus.EVALUATOR_PASSED.value == "evaluator_passed"


def test_final_result_is_strict_frozen_and_has_only_minimal_identity() -> None:
    result = FinalEvaluationResult.model_validate(values())

    assert tuple(FinalEvaluationResult.model_fields) == (
        "task_id",
        "task_fingerprint",
        "proposal_digest",
        "policy_verification_fingerprint",
        "status",
        "verification_summary",
    )
    assert set(FinalEvaluationResult.model_fields).isdisjoint(
        {
            "resolved",
            "evidence",
            "findings",
            "gold_patch",
            "workspace_root",
            "upstream_results",
            "final_evaluation_fingerprint",
        }
    )
    with pytest.raises(ValidationError):
        result.task_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        FinalEvaluationResult.model_validate({**values(), "unknown": True})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", ""),
        ("task_fingerprint", "A" * 64),
        ("proposal_digest", "b" * 63),
        ("policy_verification_fingerprint", "not-a-hash"),
        ("status", "failed"),
        ("verification_summary", "The patch is correct."),
    ],
)
def test_final_result_rejects_noncanonical_required_fields(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        FinalEvaluationResult.model_validate({**values(), field: value})


def test_final_result_requires_every_field_and_exact_system_summary() -> None:
    for field in values():
        incomplete = values()
        del incomplete[field]
        with pytest.raises(ValidationError):
            FinalEvaluationResult.model_validate(incomplete)

    result = FinalEvaluationResult.model_validate(values())
    assert result.verification_summary == (
        "The exact candidate satisfied every configured RepoFix verification gate for this task. "
        "This evaluator-scoped result is not proof of universal correctness."
    )
