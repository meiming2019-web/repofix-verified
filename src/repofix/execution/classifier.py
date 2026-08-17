"""Generic deterministic classification of one bounded command result."""

from enum import StrEnum

from repofix.reproduction.models import (
    ReproductionEvidence,
    ReproductionTerminationReason,
)


class CommandOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


def classify_command_evidence(evidence: ReproductionEvidence) -> CommandOutcome:
    """Classify exit and bounded-termination evidence without workflow semantics."""
    if evidence.termination_reason is not ReproductionTerminationReason.COMPLETED:
        return CommandOutcome.INCONCLUSIVE
    if evidence.had_decode_errors:
        return CommandOutcome.INCONCLUSIVE
    if evidence.exit_code == 0:
        return CommandOutcome.PASSED
    return CommandOutcome.FAILED
