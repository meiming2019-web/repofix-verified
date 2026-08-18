"""Strict system-owned models for evaluator-only hidden verification."""

import hashlib
import json
import re
from enum import StrEnum
from typing import Self

from pydantic import field_validator, model_validator

from repofix.execution.classifier import CommandOutcome, classify_command_evidence
from repofix.patching.models import PatchApplicationStatus
from repofix.regression.models import RegressionVerificationStatus
from repofix.reproduction.models import ReproductionEvidence
from repofix.reproduction.post_patch import PostPatchReproductionStatus
from repofix.tasks.spec import HiddenVerificationSpecification, StrictFrozenModel


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def compute_hidden_specification_fingerprint(
    specification: HiddenVerificationSpecification,
) -> str:
    """Hash the complete evaluator-only hidden specification."""
    canonical = json.dumps(
        specification.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_hidden_verification_fingerprint(
    result: "HiddenVerificationResult",
) -> str:
    """Hash the complete hidden-verification result using canonical JSON."""
    canonical = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class HiddenVerificationStatus(StrEnum):
    HIDDEN_COMMAND_PASSED = "hidden_command_passed"
    HIDDEN_COMMAND_FAILED = "hidden_command_failed"
    INCONCLUSIVE = "inconclusive"


HIDDEN_VERIFICATION_PASSED_SUMMARY = (
    "The configured evaluator-only hidden command passed for the applied candidate. "
    "Final repair evaluation has not run."
)
HIDDEN_VERIFICATION_FAILED_SUMMARY = (
    "The configured evaluator-only hidden command failed for the applied candidate. "
    "Final repair evaluation has not run."
)
HIDDEN_VERIFICATION_INCONCLUSIVE_SUMMARY = (
    "Hidden verification was inconclusive. Final repair evaluation has not run."
)

_STATUS_BY_OUTCOME = {
    CommandOutcome.PASSED: HiddenVerificationStatus.HIDDEN_COMMAND_PASSED,
    CommandOutcome.FAILED: HiddenVerificationStatus.HIDDEN_COMMAND_FAILED,
    CommandOutcome.INCONCLUSIVE: HiddenVerificationStatus.INCONCLUSIVE,
}
_SUMMARY_BY_STATUS = {
    HiddenVerificationStatus.HIDDEN_COMMAND_PASSED: (
        HIDDEN_VERIFICATION_PASSED_SUMMARY
    ),
    HiddenVerificationStatus.HIDDEN_COMMAND_FAILED: (
        HIDDEN_VERIFICATION_FAILED_SUMMARY
    ),
    HiddenVerificationStatus.INCONCLUSIVE: (
        HIDDEN_VERIFICATION_INCONCLUSIVE_SUMMARY
    ),
}


def hidden_status_for(evidence: ReproductionEvidence) -> HiddenVerificationStatus:
    return _STATUS_BY_OUTCOME[classify_command_evidence(evidence)]


def hidden_summary_for(status: HiddenVerificationStatus) -> str:
    return _SUMMARY_BY_STATUS[status]


class HiddenVerificationResult(StrictFrozenModel):
    task_id: str
    task_fingerprint: str
    hidden_specification_fingerprint: str
    reproduction_expectation_fingerprint: str
    original_reproduction_run_fingerprint: str
    proposal_digest: str
    regression_baseline_fingerprint: str
    application_result_fingerprint: str
    post_patch_reproduction_fingerprint: str
    regression_verification_fingerprint: str
    application_status: PatchApplicationStatus
    post_patch_reproduction_status: PostPatchReproductionStatus
    regression_verification_status: RegressionVerificationStatus
    command_id: str
    evidence: ReproductionEvidence
    status: HiddenVerificationStatus
    verification_summary: str

    @field_validator(
        "task_fingerprint",
        "hidden_specification_fingerprint",
        "reproduction_expectation_fingerprint",
        "original_reproduction_run_fingerprint",
        "proposal_digest",
        "regression_baseline_fingerprint",
        "application_result_fingerprint",
        "post_patch_reproduction_fingerprint",
        "regression_verification_fingerprint",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("hidden verification hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_canonical_result(self) -> Self:
        if self.application_status is not PatchApplicationStatus.APPLIED:
            raise ValueError("hidden verification requires an applied patch")
        if (
            self.post_patch_reproduction_status
            is not PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
        ):
            raise ValueError("hidden verification requires the post-patch gate")
        if (
            self.regression_verification_status
            is not RegressionVerificationStatus.REGRESSION_COMMAND_PASSED
        ):
            raise ValueError("hidden verification requires the public regression gate")
        if self.command_id != self.evidence.command_id:
            raise ValueError("hidden command identity must match its evidence")
        if self.status is not _STATUS_BY_OUTCOME[
            classify_command_evidence(self.evidence)
        ]:
            raise ValueError("hidden status does not match its evidence")
        if self.verification_summary != _SUMMARY_BY_STATUS[self.status]:
            raise ValueError("hidden result requires its canonical system summary")
        return self
