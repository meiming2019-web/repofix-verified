"""Strict system-owned models for regression baseline and verification."""

import hashlib
import json
import re
from enum import StrEnum
from typing import Self

from pydantic import field_validator, model_validator

from repofix.patching.models import PatchApplicationStatus
from repofix.regression.classifier import (
    RegressionCommandOutcome,
    classify_regression_evidence,
)
from repofix.reproduction.models import ReproductionEvidence
from repofix.reproduction.post_patch import PostPatchReproductionStatus
from repofix.tasks.spec import RegressionSpecification, StrictFrozenModel


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def compute_regression_specification_fingerprint(
    specification: RegressionSpecification,
) -> str:
    """Hash the complete evaluator-controlled regression specification."""
    canonical = json.dumps(
        specification.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RegressionBaselineStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


REGRESSION_BASELINE_PASSED_SUMMARY = (
    "The configured regression command passed before patch application. "
    "No post-patch regression or hidden verification has run."
)
REGRESSION_BASELINE_FAILED_SUMMARY = (
    "The configured regression command failed before patch application. "
    "No valid post-patch regression comparison is available."
)
REGRESSION_BASELINE_INCONCLUSIVE_SUMMARY = (
    "The pre-patch regression baseline was inconclusive. "
    "No valid post-patch regression comparison is available."
)

_BASELINE_STATUS_BY_OUTCOME = {
    RegressionCommandOutcome.PASSED: RegressionBaselineStatus.PASSED,
    RegressionCommandOutcome.FAILED: RegressionBaselineStatus.FAILED,
    RegressionCommandOutcome.INCONCLUSIVE: RegressionBaselineStatus.INCONCLUSIVE,
}
_BASELINE_SUMMARY_BY_STATUS = {
    RegressionBaselineStatus.PASSED: REGRESSION_BASELINE_PASSED_SUMMARY,
    RegressionBaselineStatus.FAILED: REGRESSION_BASELINE_FAILED_SUMMARY,
    RegressionBaselineStatus.INCONCLUSIVE: REGRESSION_BASELINE_INCONCLUSIVE_SUMMARY,
}


def baseline_status_for(evidence: ReproductionEvidence) -> RegressionBaselineStatus:
    return _BASELINE_STATUS_BY_OUTCOME[classify_regression_evidence(evidence)]


def baseline_summary_for(status: RegressionBaselineStatus) -> str:
    return _BASELINE_SUMMARY_BY_STATUS[status]


class RegressionBaselineResult(StrictFrozenModel):
    task_id: str
    task_fingerprint: str
    regression_specification_fingerprint: str
    reproduction_expectation_fingerprint: str
    original_reproduction_run_fingerprint: str
    proposal_digest: str
    command_id: str
    evidence: ReproductionEvidence
    status: RegressionBaselineStatus
    baseline_summary: str

    @field_validator(
        "task_fingerprint",
        "regression_specification_fingerprint",
        "reproduction_expectation_fingerprint",
        "original_reproduction_run_fingerprint",
        "proposal_digest",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("regression baseline hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_canonical_result(self) -> Self:
        if self.command_id != self.evidence.command_id:
            raise ValueError("regression baseline command identity must match its evidence")
        expected_status = _BASELINE_STATUS_BY_OUTCOME[
            classify_regression_evidence(self.evidence)
        ]
        if self.status is not expected_status:
            raise ValueError("regression baseline status does not match its evidence")
        if self.baseline_summary != _BASELINE_SUMMARY_BY_STATUS[self.status]:
            raise ValueError("regression baseline requires its canonical system summary")
        return self


def compute_regression_baseline_fingerprint(result: RegressionBaselineResult) -> str:
    """Hash all identity-relevant baseline result fields using canonical JSON."""
    canonical = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RegressionVerificationStatus(StrEnum):
    REGRESSION_COMMAND_PASSED = "regression_command_passed"
    REGRESSION_COMMAND_FAILED = "regression_command_failed"
    INCONCLUSIVE = "inconclusive"


REGRESSION_VERIFICATION_PASSED_SUMMARY = (
    "The configured regression command passed both before and after the patch. "
    "Hidden verification and final repair evaluation have not run."
)
REGRESSION_VERIFICATION_FAILED_SUMMARY = (
    "The configured regression command passed before the patch but failed after the patch. "
    "Hidden verification and final repair evaluation have not run."
)
REGRESSION_VERIFICATION_INCONCLUSIVE_SUMMARY = (
    "Post-patch regression verification was inconclusive. "
    "Hidden verification and final repair evaluation have not run."
)

_VERIFICATION_STATUS_BY_OUTCOME = {
    RegressionCommandOutcome.PASSED: (
        RegressionVerificationStatus.REGRESSION_COMMAND_PASSED
    ),
    RegressionCommandOutcome.FAILED: (
        RegressionVerificationStatus.REGRESSION_COMMAND_FAILED
    ),
    RegressionCommandOutcome.INCONCLUSIVE: RegressionVerificationStatus.INCONCLUSIVE,
}
_VERIFICATION_SUMMARY_BY_STATUS = {
    RegressionVerificationStatus.REGRESSION_COMMAND_PASSED: (
        REGRESSION_VERIFICATION_PASSED_SUMMARY
    ),
    RegressionVerificationStatus.REGRESSION_COMMAND_FAILED: (
        REGRESSION_VERIFICATION_FAILED_SUMMARY
    ),
    RegressionVerificationStatus.INCONCLUSIVE: (
        REGRESSION_VERIFICATION_INCONCLUSIVE_SUMMARY
    ),
}


def verification_status_for(
    evidence: ReproductionEvidence,
) -> RegressionVerificationStatus:
    return _VERIFICATION_STATUS_BY_OUTCOME[classify_regression_evidence(evidence)]


def verification_summary_for(status: RegressionVerificationStatus) -> str:
    return _VERIFICATION_SUMMARY_BY_STATUS[status]


class RegressionVerificationResult(StrictFrozenModel):
    task_id: str
    task_fingerprint: str
    regression_specification_fingerprint: str
    regression_baseline_fingerprint: str
    reproduction_expectation_fingerprint: str
    original_reproduction_run_fingerprint: str
    proposal_digest: str
    application_result_fingerprint: str
    post_patch_reproduction_fingerprint: str
    application_status: PatchApplicationStatus
    post_patch_reproduction_status: PostPatchReproductionStatus
    command_id: str
    evidence: ReproductionEvidence
    status: RegressionVerificationStatus
    verification_summary: str

    @field_validator(
        "task_fingerprint",
        "regression_specification_fingerprint",
        "regression_baseline_fingerprint",
        "reproduction_expectation_fingerprint",
        "original_reproduction_run_fingerprint",
        "proposal_digest",
        "application_result_fingerprint",
        "post_patch_reproduction_fingerprint",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("regression verification hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_canonical_result(self) -> Self:
        if self.application_status is not PatchApplicationStatus.APPLIED:
            raise ValueError("regression verification requires an applied patch")
        if (
            self.post_patch_reproduction_status
            is not PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
        ):
            raise ValueError("regression verification requires the post-patch gate")
        if self.command_id != self.evidence.command_id:
            raise ValueError("regression verification command identity must match evidence")
        expected_status = _VERIFICATION_STATUS_BY_OUTCOME[
            classify_regression_evidence(self.evidence)
        ]
        if self.status is not expected_status:
            raise ValueError("regression verification status does not match its evidence")
        if self.verification_summary != _VERIFICATION_SUMMARY_BY_STATUS[self.status]:
            raise ValueError("regression verification requires its canonical system summary")
        return self
