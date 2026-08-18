"""Strict system-owned models for terminal evaluator success."""

import re
from enum import StrEnum
from typing import Self

from pydantic import field_validator, model_validator

from repofix.tasks.spec import StrictFrozenModel


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FinalEvaluationStatus(StrEnum):
    EVALUATOR_PASSED = "evaluator_passed"


FINAL_EVALUATION_PASSED_SUMMARY = (
    "The exact candidate satisfied every configured RepoFix verification gate for this task. "
    "This evaluator-scoped result is not proof of universal correctness."
)


class FinalEvaluationResult(StrictFrozenModel):
    """Minimal terminal identity for an exact evaluator-passing candidate."""

    task_id: str
    task_fingerprint: str
    proposal_digest: str
    policy_verification_fingerprint: str
    status: FinalEvaluationStatus
    verification_summary: str

    @field_validator(
        "task_fingerprint", "proposal_digest", "policy_verification_fingerprint"
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("final evaluation hashes must be lowercase SHA-256")
        return value

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        if not value.strip() or "\0" in value:
            raise ValueError("final evaluation task ID must be nonempty and NUL-free")
        return value

    @model_validator(mode="after")
    def validate_canonical_result(self) -> Self:
        if self.status is not FinalEvaluationStatus.EVALUATOR_PASSED:
            raise ValueError("final evaluation supports only evaluator-passed results")
        if self.verification_summary != FINAL_EVALUATION_PASSED_SUMMARY:
            raise ValueError("final evaluation requires its canonical system summary")
        return self
