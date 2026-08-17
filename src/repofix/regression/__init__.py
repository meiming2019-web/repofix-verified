"""System-owned regression baseline and post-patch verification APIs."""

from repofix.regression.baseline import (
    RegressionBaselineError,
    establish_regression_baseline,
)
from repofix.regression.models import (
    REGRESSION_BASELINE_FAILED_SUMMARY,
    REGRESSION_BASELINE_INCONCLUSIVE_SUMMARY,
    REGRESSION_BASELINE_PASSED_SUMMARY,
    REGRESSION_VERIFICATION_FAILED_SUMMARY,
    REGRESSION_VERIFICATION_INCONCLUSIVE_SUMMARY,
    REGRESSION_VERIFICATION_PASSED_SUMMARY,
    RegressionBaselineResult,
    RegressionBaselineStatus,
    RegressionVerificationResult,
    RegressionVerificationStatus,
    compute_regression_baseline_fingerprint,
    compute_regression_specification_fingerprint,
)
from repofix.regression.verification import (
    RegressionVerificationError,
    verify_post_patch_regression,
)
from repofix.tasks.spec import RegressionSpecification

__all__ = [
    "REGRESSION_BASELINE_FAILED_SUMMARY",
    "REGRESSION_BASELINE_INCONCLUSIVE_SUMMARY",
    "REGRESSION_BASELINE_PASSED_SUMMARY",
    "REGRESSION_VERIFICATION_FAILED_SUMMARY",
    "REGRESSION_VERIFICATION_INCONCLUSIVE_SUMMARY",
    "REGRESSION_VERIFICATION_PASSED_SUMMARY",
    "RegressionBaselineError",
    "RegressionBaselineResult",
    "RegressionBaselineStatus",
    "RegressionSpecification",
    "RegressionVerificationError",
    "RegressionVerificationResult",
    "RegressionVerificationStatus",
    "compute_regression_baseline_fingerprint",
    "compute_regression_specification_fingerprint",
    "establish_regression_baseline",
    "verify_post_patch_regression",
]
