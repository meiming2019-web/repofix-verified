"""Deterministic evaluator-controlled reproduction verification."""

from importlib import import_module
from typing import TYPE_CHECKING

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
    compute_reproduction_expectation_fingerprint,
)
from repofix.reproduction.verifier import (
    COMBINED_OUTPUT_SEPARATOR,
    ReproductionVerificationError,
    verify_reproduction,
)

_POST_PATCH_EXPORTS = {
    "POST_PATCH_INCONCLUSIVE_SUMMARY",
    "POST_PATCH_NOT_REPRODUCED_SUMMARY",
    "POST_PATCH_STILL_REPRODUCED_SUMMARY",
    "PostPatchReproductionError",
    "PostPatchReproductionResult",
    "PostPatchReproductionStatus",
    "verify_post_patch_reproduction",
}

if TYPE_CHECKING:
    from repofix.reproduction.post_patch import (
        POST_PATCH_INCONCLUSIVE_SUMMARY,
        POST_PATCH_NOT_REPRODUCED_SUMMARY,
        POST_PATCH_STILL_REPRODUCED_SUMMARY,
        PostPatchReproductionError,
        PostPatchReproductionResult,
        PostPatchReproductionStatus,
        verify_post_patch_reproduction,
    )


def __getattr__(name: str) -> object:
    if name not in _POST_PATCH_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("repofix.reproduction.post_patch")
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "COMBINED_OUTPUT_SEPARATOR",
    "MAX_REPRODUCTION_FRAGMENT_LENGTH",
    "POST_PATCH_INCONCLUSIVE_SUMMARY",
    "POST_PATCH_NOT_REPRODUCED_SUMMARY",
    "POST_PATCH_STILL_REPRODUCED_SUMMARY",
    "PostPatchReproductionError",
    "PostPatchReproductionResult",
    "PostPatchReproductionStatus",
    "ReproductionEvidence",
    "ReproductionExpectation",
    "ReproductionOutputFragment",
    "ReproductionOutputStream",
    "ReproductionStatus",
    "ReproductionTaskBundle",
    "ReproductionTerminationReason",
    "ReproductionVerdict",
    "compute_reproduction_expectation_fingerprint",
    "ReproductionVerificationError",
    "verify_reproduction",
    "verify_post_patch_reproduction",
]
