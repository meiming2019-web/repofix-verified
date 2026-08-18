"""Strict-success-only terminal evaluation APIs."""

from repofix.final_evaluation.errors import FinalEvaluationError
from repofix.final_evaluation.finalization import finalize_evaluation
from repofix.final_evaluation.models import (
    FINAL_EVALUATION_PASSED_SUMMARY,
    FinalEvaluationResult,
    FinalEvaluationStatus,
)

__all__ = [
    "FINAL_EVALUATION_PASSED_SUMMARY",
    "FinalEvaluationError",
    "FinalEvaluationResult",
    "FinalEvaluationStatus",
    "finalize_evaluation",
]
