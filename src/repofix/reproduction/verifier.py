"""Pure deterministic verification of evaluator-controlled reproduction evidence."""

from pydantic import ValidationError

from repofix.reproduction.models import (
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionOutputFragment,
    ReproductionOutputStream,
    ReproductionStatus,
    ReproductionTerminationReason,
    ReproductionVerdict,
)


COMBINED_OUTPUT_SEPARATOR = "\n<repofix-stderr>\n"


class ReproductionVerificationError(RuntimeError):
    """Raised when evidence cannot be compared because evaluator inputs conflict."""


def _normalize_output(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _matches(
    fragment: ReproductionOutputFragment,
    *,
    stdout: str,
    stderr: str,
    combined: str,
) -> bool:
    streams = {
        ReproductionOutputStream.STDOUT: stdout,
        ReproductionOutputStream.STDERR: stderr,
        ReproductionOutputStream.COMBINED: combined,
    }
    return fragment.text in streams[fragment.stream]


def _validate_inputs(
    expectation: ReproductionExpectation, evidence: ReproductionEvidence
) -> None:
    if not isinstance(expectation, ReproductionExpectation):
        raise ReproductionVerificationError(
            "expectation must be a ReproductionExpectation"
        )
    if not isinstance(evidence, ReproductionEvidence):
        raise ReproductionVerificationError("evidence must be ReproductionEvidence")
    try:
        ReproductionExpectation.model_validate(expectation.model_dump())
        ReproductionEvidence.model_validate(evidence.model_dump())
    except ValidationError as error:
        raise ReproductionVerificationError(
            "reproduction evaluator inputs are internally inconsistent"
        ) from error
    if evidence.command_id != expectation.command_id:
        raise ReproductionVerificationError(
            "evidence command ID does not match the reproduction expectation"
        )


def verify_reproduction(
    *,
    expectation: ReproductionExpectation,
    evidence: ReproductionEvidence,
) -> ReproductionVerdict:
    """Compare bounded execution evidence with literal evaluator signatures."""
    _validate_inputs(expectation, evidence)
    stdout = _normalize_output(evidence.stdout)
    stderr = _normalize_output(evidence.stderr)
    combined = stdout + COMBINED_OUTPUT_SEPARATOR + stderr

    matched_required = tuple(
        sorted(
            fragment.fragment_id
            for fragment in expectation.required_fragments
            if _matches(fragment, stdout=stdout, stderr=stderr, combined=combined)
        )
    )
    missing_required = tuple(
        sorted(
            fragment.fragment_id
            for fragment in expectation.required_fragments
            if not _matches(fragment, stdout=stdout, stderr=stderr, combined=combined)
        )
    )
    forbidden_found = tuple(
        sorted(
            fragment.fragment_id
            for fragment in expectation.forbidden_fragments
            if _matches(fragment, stdout=stdout, stderr=stderr, combined=combined)
        )
    )

    reasons: tuple[str, ...]
    if evidence.termination_reason is ReproductionTerminationReason.TIMED_OUT:
        status = ReproductionStatus.INCONCLUSIVE
        reasons = ("command execution timed out",)
    elif evidence.termination_reason is ReproductionTerminationReason.OUTPUT_LIMIT:
        status = ReproductionStatus.INCONCLUSIVE
        reasons = ("command output limit was exceeded",)
    elif evidence.had_decode_errors:
        status = ReproductionStatus.INCONCLUSIVE
        reasons = ("command output contained UTF-8 decoding errors",)
    elif forbidden_found:
        status = ReproductionStatus.INCONCLUSIVE
        reasons = ("forbidden output was present",)
    elif evidence.exit_code == 0:
        status = ReproductionStatus.NOT_REPRODUCED
        reasons = ("command completed with exit code zero",)
    elif evidence.exit_code not in expectation.expected_exit_codes:
        status = ReproductionStatus.INCONCLUSIVE
        reasons = ("command exited with an unexpected nonzero exit code",)
    elif missing_required:
        status = ReproductionStatus.INCONCLUSIVE
        reasons = ("required reproduction output was missing",)
    else:
        status = ReproductionStatus.REPRODUCED
        reasons = ("expected failing behavior was reproduced",)

    return ReproductionVerdict(
        status=status,
        command_id=evidence.command_id,
        exit_code=evidence.exit_code,
        reasons=reasons,
        matched_required_fragment_ids=matched_required,
        missing_required_fragment_ids=missing_required,
        forbidden_fragment_ids_found=forbidden_found,
    )
