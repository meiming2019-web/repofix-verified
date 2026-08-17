"""Shared deterministic classification of one bounded regression command."""

from enum import StrEnum

from repofix.reproduction.models import (
    ReproductionEvidence,
    ReproductionTerminationReason,
)


class RegressionCommandOutcome(StrEnum):
    """Execution-only meaning shared by baseline and post-patch stages."""

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


def classify_regression_evidence(
    evidence: ReproductionEvidence,
) -> RegressionCommandOutcome:
    """Classify bounded command evidence without reproduction semantics."""
    if evidence.termination_reason is not ReproductionTerminationReason.COMPLETED:
        return RegressionCommandOutcome.INCONCLUSIVE
    if evidence.had_decode_errors:
        return RegressionCommandOutcome.INCONCLUSIVE
    if evidence.exit_code == 0:
        return RegressionCommandOutcome.PASSED
    return RegressionCommandOutcome.FAILED
