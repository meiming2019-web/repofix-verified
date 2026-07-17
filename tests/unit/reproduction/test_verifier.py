"""Tests for the pure deterministic reproduction verifier."""

from copy import deepcopy

import pytest

from repofix.reproduction import (
    COMBINED_OUTPUT_SEPARATOR,
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionOutputFragment,
    ReproductionOutputStream,
    ReproductionStatus,
    ReproductionTerminationReason,
    ReproductionVerificationError,
    verify_reproduction,
)


def output_fragment(
    fragment_id: str,
    text: str,
    stream: ReproductionOutputStream = ReproductionOutputStream.COMBINED,
) -> ReproductionOutputFragment:
    return ReproductionOutputFragment(
        fragment_id=fragment_id,
        stream=stream,
        text=text,
    )


def expectation(
    *,
    required: tuple[ReproductionOutputFragment, ...] | None = None,
    forbidden: tuple[ReproductionOutputFragment, ...] = (),
    exit_codes: tuple[int, ...] = (1,),
) -> ReproductionExpectation:
    return ReproductionExpectation(
        command_id="unit_tests",
        expected_exit_codes=exit_codes,
        required_fragments=required
        or (
            output_fragment("target-test", "test_empty_header_retains_configured_value"),
            output_fragment("target-assertion", "assert 'default' == 'configured'"),
            output_fragment("failure-count", "1 failed, 1 passed"),
        ),
        forbidden_fragments=forbidden,
    )


def evidence(
    *,
    stdout: str = (
        "test_empty_header_retains_configured_value\n"
        "assert 'default' == 'configured'\n"
        "1 failed, 1 passed\n"
    ),
    stderr: str = "",
    termination_reason: ReproductionTerminationReason = (
        ReproductionTerminationReason.COMPLETED
    ),
    exit_code: int | None = 1,
    had_decode_errors: bool = False,
    command_id: str = "unit_tests",
) -> ReproductionEvidence:
    return ReproductionEvidence(
        command_id=command_id,
        argv=("pytest", "-q"),
        termination_reason=termination_reason,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdout_bytes=len(stdout.encode("utf-8")),
        stderr_bytes=len(stderr.encode("utf-8")),
        had_decode_errors=had_decode_errors,
    )


def test_fixture_style_evidence_is_reproduced() -> None:
    verdict = verify_reproduction(expectation=expectation(), evidence=evidence())

    assert verdict.status is ReproductionStatus.REPRODUCED
    assert verdict.matched_required_fragment_ids == (
        "failure-count",
        "target-assertion",
        "target-test",
    )
    assert verdict.missing_required_fragment_ids == ()
    assert verdict.forbidden_fragment_ids_found == ()


@pytest.mark.parametrize(
    "stdout",
    [
        "unrelated assertion failed\n1 failed\n",
        "AssertionError: unrelated values differ\n",
    ],
    ids=["missing-target-fragments", "unrelated-assertion"],
)
def test_exit_one_unrelated_failure_is_inconclusive(stdout: str) -> None:
    verdict = verify_reproduction(
        expectation=expectation(), evidence=evidence(stdout=stdout)
    )

    assert verdict.status is ReproductionStatus.INCONCLUSIVE
    assert verdict.missing_required_fragment_ids


@pytest.mark.parametrize(
    ("failure_text", "fragment_id"),
    [
        ("ModuleNotFoundError: No module named 'header_parser'", "import-error"),
        ("ERROR collecting tests/test_header_parser.py", "collection-error"),
    ],
)
def test_forbidden_import_or_collection_failure_is_inconclusive(
    failure_text: str, fragment_id: str
) -> None:
    rule = output_fragment(fragment_id, failure_text)
    verdict = verify_reproduction(
        expectation=expectation(forbidden=(rule,)),
        evidence=evidence(stdout=evidence().stdout + failure_text),
    )

    assert verdict.status is ReproductionStatus.INCONCLUSIVE
    assert verdict.forbidden_fragment_ids_found == (fragment_id,)


def test_exit_zero_is_not_reproduced() -> None:
    verdict = verify_reproduction(
        expectation=expectation(), evidence=evidence(exit_code=0)
    )

    assert verdict.status is ReproductionStatus.NOT_REPRODUCED


def test_forbidden_fragment_makes_exit_zero_inconclusive() -> None:
    rule = output_fragment("collection-error", "ERROR collecting")
    verdict = verify_reproduction(
        expectation=expectation(forbidden=(rule,)),
        evidence=evidence(stdout="ERROR collecting", exit_code=0),
    )

    assert verdict.status is ReproductionStatus.INCONCLUSIVE
    assert verdict.forbidden_fragment_ids_found == ("collection-error",)


@pytest.mark.parametrize(
    "termination_reason",
    [
        ReproductionTerminationReason.TIMED_OUT,
        ReproductionTerminationReason.OUTPUT_LIMIT,
    ],
)
def test_bounded_termination_is_inconclusive(
    termination_reason: ReproductionTerminationReason,
) -> None:
    verdict = verify_reproduction(
        expectation=expectation(),
        evidence=evidence(termination_reason=termination_reason, exit_code=None),
    )

    assert verdict.status is ReproductionStatus.INCONCLUSIVE


def test_decode_errors_are_inconclusive() -> None:
    verdict = verify_reproduction(
        expectation=expectation(), evidence=evidence(had_decode_errors=True)
    )

    assert verdict.status is ReproductionStatus.INCONCLUSIVE


def test_unexpected_nonzero_exit_code_is_inconclusive() -> None:
    verdict = verify_reproduction(
        expectation=expectation(), evidence=evidence(exit_code=2)
    )

    assert verdict.status is ReproductionStatus.INCONCLUSIVE
    assert "unexpected nonzero" in verdict.reasons[0]


def test_missing_one_required_fragment_is_inconclusive() -> None:
    partial = evidence().stdout.replace("1 failed, 1 passed\n", "")
    verdict = verify_reproduction(
        expectation=expectation(), evidence=evidence(stdout=partial)
    )

    assert verdict.status is ReproductionStatus.INCONCLUSIVE
    assert verdict.missing_required_fragment_ids == ("failure-count",)


def test_stream_specific_and_combined_matching() -> None:
    rules = (
        output_fragment("stdout-rule", "stdout target", ReproductionOutputStream.STDOUT),
        output_fragment("stderr-rule", "stderr target", ReproductionOutputStream.STDERR),
        output_fragment(
            "combined-rule", "combined visible", ReproductionOutputStream.COMBINED
        ),
    )
    verdict = verify_reproduction(
        expectation=expectation(required=rules),
        evidence=evidence(
            stdout="stdout target\ncombined visible", stderr="stderr target"
        ),
    )

    assert COMBINED_OUTPUT_SEPARATOR == "\n<repofix-stderr>\n"
    assert verdict.status is ReproductionStatus.REPRODUCED
    assert verdict.matched_required_fragment_ids == (
        "combined-rule",
        "stderr-rule",
        "stdout-rule",
    )


def test_combined_fragment_cannot_match_across_stream_boundary() -> None:
    rule = output_fragment("cross-boundary", "stdout end\nstderr start")
    verdict = verify_reproduction(
        expectation=expectation(required=(rule,)),
        evidence=evidence(stdout="stdout end", stderr="stderr start"),
    )

    assert verdict.status is ReproductionStatus.INCONCLUSIVE
    assert verdict.missing_required_fragment_ids == ("cross-boundary",)


def test_stream_specific_fragments_do_not_match_the_other_stream() -> None:
    rule = output_fragment(
        "stdout-only", "only in stderr", ReproductionOutputStream.STDOUT
    )
    verdict = verify_reproduction(
        expectation=expectation(required=(rule,)),
        evidence=evidence(stdout="", stderr="only in stderr"),
    )

    assert verdict.status is ReproductionStatus.INCONCLUSIVE
    assert verdict.missing_required_fragment_ids == ("stdout-only",)


def test_matching_is_case_sensitive_and_literal_not_regex() -> None:
    rules = (
        output_fragment("case-sensitive", "Target Failure"),
        output_fragment("literal-meta", "value.*[fixed]"),
    )
    wrong_case = verify_reproduction(
        expectation=expectation(required=rules),
        evidence=evidence(stdout="target failure\nvalue other fixed\n"),
    )
    literal = verify_reproduction(
        expectation=expectation(required=rules),
        evidence=evidence(stdout="Target Failure\nvalue.*[fixed]\n"),
    )

    assert wrong_case.status is ReproductionStatus.INCONCLUSIVE
    assert literal.status is ReproductionStatus.REPRODUCED


def test_crlf_and_remaining_carriage_returns_are_normalized() -> None:
    rules = (output_fragment("normalized", "first\nsecond\nthird"),)
    verdict = verify_reproduction(
        expectation=expectation(required=rules),
        evidence=evidence(stdout="first\r\nsecond\rthird"),
    )

    assert verdict.status is ReproductionStatus.REPRODUCED


def test_command_id_mismatch_and_wrong_evidence_type_raise() -> None:
    with pytest.raises(ReproductionVerificationError, match="command ID"):
        verify_reproduction(
            expectation=expectation(), evidence=evidence(command_id="other_tests")
        )
    with pytest.raises(ReproductionVerificationError, match="evidence"):
        verify_reproduction(
            expectation=expectation(),
            evidence=object(),  # type: ignore[arg-type]
        )


def test_fragment_id_output_order_is_deterministic() -> None:
    required = (
        output_fragment("z-required", "missing z"),
        output_fragment("a-required", "matched a"),
        output_fragment("m-required", "missing m"),
    )
    forbidden = (
        output_fragment("z-forbidden", "found z"),
        output_fragment("a-forbidden", "found a"),
    )
    verdict = verify_reproduction(
        expectation=expectation(required=required, forbidden=forbidden),
        evidence=evidence(stdout="matched a\nfound z\nfound a"),
    )

    assert verdict.matched_required_fragment_ids == ("a-required",)
    assert verdict.missing_required_fragment_ids == ("m-required", "z-required")
    assert verdict.forbidden_fragment_ids_found == ("a-forbidden", "z-forbidden")


def test_verification_does_not_mutate_evidence_or_copy_output_to_verdict() -> None:
    value = evidence(stdout="distinctive untrusted full output\n")
    before = deepcopy(value.model_dump())

    verdict = verify_reproduction(expectation=expectation(), evidence=value)

    assert value.model_dump() == before
    assert "distinctive untrusted full output" not in repr(verdict.model_dump())
