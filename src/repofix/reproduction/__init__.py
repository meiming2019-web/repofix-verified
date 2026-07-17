"""Deterministic evaluator-controlled reproduction verification."""

from repofix.reproduction.models import (
    MAX_REPRODUCTION_FRAGMENT_LENGTH,
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionOutputFragment,
    ReproductionOutputStream,
    ReproductionStatus,
    ReproductionTaskBundle,
    ReproductionTerminationReason,
    ReproductionVerdict,
)
from repofix.reproduction.verifier import (
    COMBINED_OUTPUT_SEPARATOR,
    ReproductionVerificationError,
    verify_reproduction,
)

__all__ = [
    "COMBINED_OUTPUT_SEPARATOR",
    "MAX_REPRODUCTION_FRAGMENT_LENGTH",
    "ReproductionEvidence",
    "ReproductionExpectation",
    "ReproductionOutputFragment",
    "ReproductionOutputStream",
    "ReproductionStatus",
    "ReproductionTaskBundle",
    "ReproductionTerminationReason",
    "ReproductionVerdict",
    "ReproductionVerificationError",
    "verify_reproduction",
]
