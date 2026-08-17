"""Evaluator-only hidden verification APIs."""

from repofix.hidden.assets import ResolvedHiddenCommand, resolve_hidden_command
from repofix.hidden.errors import HiddenVerificationError
from repofix.hidden.interfaces import EvaluatorCommandGateway
from repofix.hidden.models import (
    HIDDEN_VERIFICATION_FAILED_SUMMARY,
    HIDDEN_VERIFICATION_INCONCLUSIVE_SUMMARY,
    HIDDEN_VERIFICATION_PASSED_SUMMARY,
    HiddenVerificationResult,
    HiddenVerificationStatus,
    compute_hidden_specification_fingerprint,
)
from repofix.hidden.verification import verify_hidden_behavior
from repofix.tasks.spec import (
    EvaluatorFileReference,
    HiddenVerificationSpecification,
)

__all__ = [
    "EvaluatorFileReference",
    "EvaluatorCommandGateway",
    "HIDDEN_VERIFICATION_FAILED_SUMMARY",
    "HIDDEN_VERIFICATION_INCONCLUSIVE_SUMMARY",
    "HIDDEN_VERIFICATION_PASSED_SUMMARY",
    "HiddenVerificationError",
    "HiddenVerificationResult",
    "HiddenVerificationSpecification",
    "HiddenVerificationStatus",
    "ResolvedHiddenCommand",
    "compute_hidden_specification_fingerprint",
    "resolve_hidden_command",
    "verify_hidden_behavior",
]
