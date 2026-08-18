"""Errors raised by strict-success-only final evaluation."""


class FinalEvaluationError(RuntimeError):
    """Raised when final evaluation cannot validly consume the supplied artifact chain."""
