"""Tests for exactly-once pre-patch regression baseline establishment."""

from pathlib import Path

import pytest

from repofix.execution import (
    ApprovedCommandExecutionError,
    CommandTerminationReason,
)
from repofix.regression import (
    REGRESSION_BASELINE_FAILED_SUMMARY,
    REGRESSION_BASELINE_INCONCLUSIVE_SUMMARY,
    REGRESSION_BASELINE_PASSED_SUMMARY,
    RegressionBaselineError,
    RegressionBaselineStatus,
    RegressionSpecification,
    establish_regression_baseline,
)

from .conftest import Gateway, PreparedBaseline, execution


def run_baseline(
    prepared: PreparedBaseline,
    gateway: Gateway,
    **updates: object,
):
    values = {
        "workspace_root": prepared.workspace,
        "task": prepared.task,
        "reproduction_expectation": prepared.expectation,
        "regression_specification": prepared.specification,
        "original_reproduction_result": prepared.original,
        "proposal": prepared.proposal,
        "command_gateway": gateway,
    }
    values.update(updates)
    return establish_regression_baseline(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("exit_code", "termination", "decode_errors", "status", "summary"),
    [
        (
            0,
            CommandTerminationReason.COMPLETED,
            False,
            RegressionBaselineStatus.PASSED,
            REGRESSION_BASELINE_PASSED_SUMMARY,
        ),
        (
            3,
            CommandTerminationReason.COMPLETED,
            False,
            RegressionBaselineStatus.FAILED,
            REGRESSION_BASELINE_FAILED_SUMMARY,
        ),
        (
            None,
            CommandTerminationReason.TIMED_OUT,
            False,
            RegressionBaselineStatus.INCONCLUSIVE,
            REGRESSION_BASELINE_INCONCLUSIVE_SUMMARY,
        ),
        (
            None,
            CommandTerminationReason.OUTPUT_LIMIT,
            False,
            RegressionBaselineStatus.INCONCLUSIVE,
            REGRESSION_BASELINE_INCONCLUSIVE_SUMMARY,
        ),
        (
            0,
            CommandTerminationReason.COMPLETED,
            True,
            RegressionBaselineStatus.INCONCLUSIVE,
            REGRESSION_BASELINE_INCONCLUSIVE_SUMMARY,
        ),
    ],
)
def test_baseline_classifies_once_with_fixed_summary(
    prepared_baseline: PreparedBaseline,
    exit_code: int | None,
    termination: CommandTerminationReason,
    decode_errors: bool,
    status: RegressionBaselineStatus,
    summary: str,
) -> None:
    argv = prepared_baseline.task.approved_commands["regression_tests"].argv
    gateway = Gateway(
        execution(
            "regression_tests",
            argv,
            exit_code=exit_code,
            termination=termination,
            decode_errors=decode_errors,
        )
    )
    before = prepared_baseline.source.read_bytes()

    result = run_baseline(prepared_baseline, gateway)

    assert result.status is status
    assert result.baseline_summary == summary
    assert gateway.calls == ["regression_tests"]
    assert prepared_baseline.source.read_bytes() == before


def test_generic_api_allows_same_reproduction_and_regression_command(
    prepared_baseline: PreparedBaseline,
) -> None:
    specification = RegressionSpecification(command_id="unit_tests")
    argv = prepared_baseline.task.approved_commands["unit_tests"].argv
    gateway = Gateway(execution("unit_tests", argv))

    result = run_baseline(
        prepared_baseline,
        gateway,
        regression_specification=specification,
    )

    assert result.status is RegressionBaselineStatus.PASSED
    assert gateway.calls == ["unit_tests"]


@pytest.mark.parametrize("mismatch", ["task", "expectation", "run", "proposal"])
def test_invalid_provenance_rejects_before_execution(
    prepared_baseline: PreparedBaseline,
    mismatch: str,
) -> None:
    gateway = Gateway(
        execution(
            "regression_tests",
            prepared_baseline.task.approved_commands["regression_tests"].argv,
        )
    )
    updates: dict[str, object]
    if mismatch == "task":
        updates = {"task": prepared_baseline.task.model_copy(update={"issue_body": "changed"})}
    elif mismatch == "expectation":
        updates = {
            "reproduction_expectation": prepared_baseline.expectation.model_copy(
                update={"expected_exit_codes": (2,)}
            )
        }
    elif mismatch == "run":
        updates = {
            "original_reproduction_result": prepared_baseline.original.model_copy(
                update={"task_fingerprint": "f" * 64}
            )
        }
    else:
        updates = {
            "proposal": prepared_baseline.proposal.model_copy(
                update={"proposal_digest": "f" * 64}
            )
        }

    with pytest.raises(RegressionBaselineError):
        run_baseline(prepared_baseline, gateway, **updates)

    assert gateway.calls == []


def test_unknown_regression_command_rejects_before_execution(
    prepared_baseline: PreparedBaseline,
) -> None:
    gateway = Gateway(None)
    specification = RegressionSpecification(command_id="missing")

    with pytest.raises(RegressionBaselineError, match="not approved"):
        run_baseline(
            prepared_baseline,
            gateway,
            regression_specification=specification,
        )

    assert gateway.calls == []


@pytest.mark.parametrize("mismatch", ["command", "argv"])
def test_gateway_identity_mismatch_is_rejected(
    prepared_baseline: PreparedBaseline,
    mismatch: str,
) -> None:
    command_id = "unit_tests" if mismatch == "command" else "regression_tests"
    argv = (
        ("unexpected",)
        if mismatch == "argv"
        else prepared_baseline.task.approved_commands["regression_tests"].argv
    )
    gateway = Gateway(execution(command_id, argv))

    with pytest.raises(RegressionBaselineError, match="inconsistent"):
        run_baseline(prepared_baseline, gateway)

    assert gateway.calls == ["regression_tests"]


def test_stale_workspace_rejects_before_execution(
    prepared_baseline: PreparedBaseline,
) -> None:
    prepared_baseline.source.write_bytes(b"changed\n")
    gateway = Gateway(None)

    with pytest.raises(RegressionBaselineError, match="proposal originals"):
        run_baseline(prepared_baseline, gateway)

    assert gateway.calls == []


def test_command_time_target_mutation_returns_no_baseline(
    prepared_baseline: PreparedBaseline,
) -> None:
    gateway = Gateway(
        execution(
            "regression_tests",
            prepared_baseline.task.approved_commands["regression_tests"].argv,
        ),
        mutation=(prepared_baseline.source, b"mutated\n"),
    )

    with pytest.raises(RegressionBaselineError, match="modified a proposal target"):
        run_baseline(prepared_baseline, gateway)

    assert gateway.calls == ["regression_tests"]


def test_operational_executor_error_propagates_without_retry(
    prepared_baseline: PreparedBaseline,
) -> None:
    error = ApprovedCommandExecutionError("bounded executor failed")
    gateway = Gateway(error=error)

    with pytest.raises(ApprovedCommandExecutionError) as caught:
        run_baseline(prepared_baseline, gateway)

    assert caught.value is error
    assert gateway.calls == ["regression_tests"]


def test_missing_workspace_is_sanitized_before_execution(
    prepared_baseline: PreparedBaseline,
    tmp_path: Path,
) -> None:
    gateway = Gateway(None)
    missing = tmp_path / "private-missing"

    with pytest.raises(RegressionBaselineError, match="could not be resolved") as caught:
        run_baseline(prepared_baseline, gateway, workspace_root=missing)

    assert str(missing) not in str(caught.value)
    assert gateway.calls == []
