"""Strict system-owned models for protected workspace policy verification."""

import hashlib
import json
import re
from enum import StrEnum
from typing import Self

from pydantic import field_validator, model_validator

from repofix.hidden.models import HiddenVerificationStatus
from repofix.patching.models import PatchApplicationStatus
from repofix.regression.models import RegressionVerificationStatus
from repofix.reproduction.post_patch import PostPatchReproductionStatus
from repofix.tasks.spec import (
    PatchPolicySpecification,
    StrictFrozenModel,
    validate_relative_source_path,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def compute_patch_policy_specification_fingerprint(
    specification: PatchPolicySpecification,
) -> str:
    """Hash the complete evaluator-only policy specification."""
    canonical = json.dumps(
        specification.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PolicyRuleId(StrEnum):
    PROTECTED_FILE_INTEGRITY = "protected_file_integrity"
    FORBIDDEN_PATH_PRESENT = "forbidden_path_present"


class PolicyVerificationStatus(StrEnum):
    POLICY_PASSED = "policy_passed"
    POLICY_FAILED = "policy_failed"


PROTECTED_FILE_INTEGRITY_SUMMARY = (
    "Protected workspace file does not match its evaluator-declared identity."
)
FORBIDDEN_PATH_PRESENT_SUMMARY = "Forbidden workspace path is present."
POLICY_VERIFICATION_PASSED_SUMMARY = (
    "The applied candidate preserved all evaluator-declared protected workspace surfaces. "
    "Final repair evaluation has not run."
)
POLICY_VERIFICATION_FAILED_SUMMARY = (
    "The applied candidate violated one or more evaluator-declared protected workspace surfaces. "
    "Final repair evaluation has not run."
)

_FINDING_SUMMARY_BY_RULE = {
    PolicyRuleId.PROTECTED_FILE_INTEGRITY: PROTECTED_FILE_INTEGRITY_SUMMARY,
    PolicyRuleId.FORBIDDEN_PATH_PRESENT: FORBIDDEN_PATH_PRESENT_SUMMARY,
}
_RESULT_SUMMARY_BY_STATUS = {
    PolicyVerificationStatus.POLICY_PASSED: POLICY_VERIFICATION_PASSED_SUMMARY,
    PolicyVerificationStatus.POLICY_FAILED: POLICY_VERIFICATION_FAILED_SUMMARY,
}


class PolicyFinding(StrictFrozenModel):
    """One sanitized deterministic protected-workspace policy violation."""

    rule_id: PolicyRuleId
    path: str
    summary: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_relative_source_path(value, description="policy finding path")

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.summary != _FINDING_SUMMARY_BY_RULE[self.rule_id]:
            raise ValueError("policy findings require their canonical system summary")
        return self


class PolicyVerificationResult(StrictFrozenModel):
    """Deterministic result of inspecting evaluator-declared workspace surfaces."""

    task_id: str
    task_fingerprint: str
    policy_specification_fingerprint: str
    proposal_digest: str
    application_result_fingerprint: str
    post_patch_reproduction_fingerprint: str
    regression_verification_fingerprint: str
    hidden_verification_fingerprint: str
    application_status: PatchApplicationStatus
    post_patch_reproduction_status: PostPatchReproductionStatus
    regression_verification_status: RegressionVerificationStatus
    hidden_verification_status: HiddenVerificationStatus
    status: PolicyVerificationStatus
    findings: tuple[PolicyFinding, ...]
    verification_summary: str

    @field_validator(
        "task_fingerprint",
        "policy_specification_fingerprint",
        "proposal_digest",
        "application_result_fingerprint",
        "post_patch_reproduction_fingerprint",
        "regression_verification_fingerprint",
        "hidden_verification_fingerprint",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("policy verification hashes must be lowercase SHA-256")
        return value

    @field_validator("findings", mode="before")
    @classmethod
    def normalize_findings(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("policy findings must be a list or tuple")
        return tuple(value)

    @model_validator(mode="after")
    def validate_canonical_result(self) -> Self:
        keys = tuple((item.rule_id.value, item.path) for item in self.findings)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("policy findings must be sorted and unique")
        expected_status = (
            PolicyVerificationStatus.POLICY_FAILED
            if self.findings
            else PolicyVerificationStatus.POLICY_PASSED
        )
        if self.status is not expected_status:
            raise ValueError("policy status must match the presence of findings")
        if self.verification_summary != _RESULT_SUMMARY_BY_STATUS[self.status]:
            raise ValueError("policy result requires its canonical system summary")
        if self.application_status is not PatchApplicationStatus.APPLIED:
            raise ValueError("policy verification requires an applied patch")
        if (
            self.post_patch_reproduction_status
            is not PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
        ):
            raise ValueError("policy verification requires the post-patch gate")
        if (
            self.regression_verification_status
            is not RegressionVerificationStatus.REGRESSION_COMMAND_PASSED
        ):
            raise ValueError("policy verification requires the public regression gate")
        if (
            self.hidden_verification_status
            is not HiddenVerificationStatus.HIDDEN_COMMAND_PASSED
        ):
            raise ValueError("policy verification requires the hidden pass gate")
        return self


def finding_summary_for(rule_id: PolicyRuleId) -> str:
    return _FINDING_SUMMARY_BY_RULE[rule_id]


def policy_summary_for(status: PolicyVerificationStatus) -> str:
    return _RESULT_SUMMARY_BY_STATUS[status]
