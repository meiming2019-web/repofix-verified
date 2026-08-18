"""Protected workspace policy verification APIs."""

from repofix.policy.errors import PolicyVerificationError
from repofix.policy.models import (
    FORBIDDEN_PATH_PRESENT_SUMMARY,
    POLICY_VERIFICATION_FAILED_SUMMARY,
    POLICY_VERIFICATION_PASSED_SUMMARY,
    PROTECTED_FILE_INTEGRITY_SUMMARY,
    PolicyFinding,
    PolicyRuleId,
    PolicyVerificationResult,
    PolicyVerificationStatus,
    compute_patch_policy_specification_fingerprint,
    compute_policy_verification_fingerprint,
)
from repofix.policy.verification import verify_patch_policy
from repofix.tasks.spec import PatchPolicySpecification, WorkspaceFileReference

__all__ = [
    "FORBIDDEN_PATH_PRESENT_SUMMARY",
    "POLICY_VERIFICATION_FAILED_SUMMARY",
    "POLICY_VERIFICATION_PASSED_SUMMARY",
    "PROTECTED_FILE_INTEGRITY_SUMMARY",
    "PatchPolicySpecification",
    "PolicyFinding",
    "PolicyRuleId",
    "PolicyVerificationError",
    "PolicyVerificationResult",
    "PolicyVerificationStatus",
    "WorkspaceFileReference",
    "compute_patch_policy_specification_fingerprint",
    "compute_policy_verification_fingerprint",
    "verify_patch_policy",
]
