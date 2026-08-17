"""Tests for gated exactly-once post-patch regression verification."""

import pytest

from repofix.execution import (
    ApprovedCommandExecutionError,
    ApprovedCommandExecutionResult,
    CommandTerminationReason,
)
from repofix.regression import (
    REGRESSION_BASELINE_FAILED_SUMMARY,
    REGRESSION_VERIFICATION_FAILED_SUMMARY,
    REGRESSION_VERIFICATION_INCONCLUSIVE_SUMMARY,
    REGRESSION_VERIFICATION_PASSED_SUMMARY,
    RegressionBaselineResult,
    RegressionBaselineStatus,
    RegressionVerificationError,
    RegressionVerificationStatus,
    verify_post_patch_regression,
)
from repofix.reproduction import (
    ReproductionEvidence,
    ReproductionStatus,
    verify_post_patch_reproduction,
)

from .conftest import Gateway, PreparedVerification, execution


def run_verification(
    prepared: PreparedVerification,
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
        "baseline_result": prepared.baseline,
        "application_result": prepared.application,
        "post_patch_reproduction_result": prepared.post_patch,
        "command_gateway": gateway,
    }
    values.update(updates)
    return verify_post_patch_regression(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("exit_code", "termination", "decode_errors", "status", "summary"),
    [
        (
            0,
            CommandTerminationReason.COMPLETED,
            False,
            RegressionVerificationStatus.REGRESSION_COMMAND_PASSED,
            REGRESSION_VERIFICATION_PASSED_SUMMARY,
        ),
        (
            2,
            CommandTerminationReason.COMPLETED,
            False,
            RegressionVerificationStatus.REGRESSION_COMMAND_FAILED,
            REGRESSION_VERIFICATION_FAILED_SUMMARY,
        ),
        (
            None,
            CommandTerminationReason.TIMED_OUT,
            False,
            RegressionVerificationStatus.INCONCLUSIVE,
            REGRESSION_VERIFICATION_INCONCLUSIVE_SUMMARY,
        ),
        (
            None,
            CommandTerminationReason.OUTPUT_LIMIT,
            False,
            RegressionVerificationStatus.INCONCLUSIVE,
            REGRESSION_VERIFICATION_INCONCLUSIVE_SUMMARY,
        ),
        (
            0,
            CommandTerminationReason.COMPLETED,
            True,
            RegressionVerificationStatus.INCONCLUSIVE,
            REGRESSION_VERIFICATION_INCONCLUSIVE_SUMMARY,
        ),
    ],
)
def test_post_patch_classifies_once_with_conservative_summary(
    prepared_verification: PreparedVerification,
    exit_code: int | None,
    termination: CommandTerminationReason,
    decode_errors: bool,
    status: RegressionVerificationStatus,
    summary: str,
) -> None:
    argv = prepared_verification.task.approved_commands["regression_tests"].argv
    gateway = Gateway(
        execution(
            "regression_tests",
            argv,
            exit_code=exit_code,
            termination=termination,
            decode_errors=decode_errors,
        )
    )
    before = prepared_verification.source.read_bytes()

    result = run_verification(prepared_verification, gateway)

    assert result.status is status
    assert result.verification_summary == summary
    assert gateway.calls == ["regression_tests"]
    assert prepared_verification.source.read_bytes() == before
    assert "repair succeeded" not in result.verification_summary.lower()
    assert "regression-free" not in result.verification_summary.lower()


def _nonpassing_baseline(
    prepared: PreparedVerification,
) -> RegressionBaselineResult:
    evidence = ReproductionEvidence.from_execution_result(
        execution(
            "regression_tests",
            prepared.task.approved_commands["regression_tests"].argv,
            exit_code=1,
        )
    )
    return RegressionBaselineResult.model_validate(
        {
            **prepared.baseline.model_dump(),
            "evidence": evidence.model_dump(),
            "status": RegressionBaselineStatus.FAILED,
            "baseline_summary": REGRESSION_BASELINE_FAILED_SUMMARY,
        }
    )


def _post_patch_result(
    prepared: PreparedVerification,
    status: ReproductionStatus,
):
    argv = prepared.task.approved_commands["unit_tests"].argv
    if status is ReproductionStatus.REPRODUCED:
        output = "TARGET FAILURE\n"
        command_result = ApprovedCommandExecutionResult(
            command_id="unit_tests",
            argv=argv,
            termination_reason=CommandTerminationReason.COMPLETED,
            exit_code=1,
            stdout=output,
            stderr="",
            stdout_bytes=len(output),
            stderr_bytes=0,
            had_decode_errors=False,
        )
    else:
        command_result = execution(
            "unit_tests",
            argv,
            exit_code=None,
            termination=CommandTerminationReason.TIMED_OUT,
        )
    return verify_post_patch_reproduction(
        workspace_root=prepared.workspace,
        task=prepared.task,
        expectation=prepared.expectation,
        original_reproduction_result=prepared.original,
        proposal=prepared.proposal,
        application_result=prepared.application,
        command_gateway=Gateway(command_result),
    )


def test_nonpassing_baseline_blocks_before_execution(
    prepared_verification: PreparedVerification,
) -> None:
    gateway = Gateway(None)

    with pytest.raises(RegressionVerificationError, match="passing baseline"):
        run_verification(
            prepared_verification,
            gateway,
            baseline_result=_nonpassing_baseline(prepared_verification),
        )

    assert gateway.calls == []


@pytest.mark.parametrize(
    "status", [ReproductionStatus.REPRODUCED, ReproductionStatus.INCONCLUSIVE]
)
def test_post_patch_reproduction_gate_blocks_before_execution(
    prepared_verification: PreparedVerification,
    status: ReproductionStatus,
) -> None:
    post_patch = _post_patch_result(prepared_verification, status)
    gateway = Gateway(None)

    with pytest.raises(RegressionVerificationError, match="must be absent"):
        run_verification(
            prepared_verification,
            gateway,
            post_patch_reproduction_result=post_patch,
        )

    assert gateway.calls == []


@pytest.mark.parametrize(
    "artifact", ["task", "baseline", "application", "post_patch"]
)
def test_stale_chain_bindings_block_before_execution(
    prepared_verification: PreparedVerification,
    artifact: str,
) -> None:
    gateway = Gateway(None)
    if artifact == "task":
        updates = {"task": prepared_verification.task.model_copy(update={"issue_body": "x"})}
    elif artifact == "baseline":
        updates = {
            "baseline_result": prepared_verification.baseline.model_copy(
                update={"proposal_digest": "f" * 64}
            )
        }
    elif artifact == "application":
        updates = {
            "application_result": prepared_verification.application.model_copy(
                update={"proposal_digest": "f" * 64}
            )
        }
    else:
        updates = {
            "post_patch_reproduction_result": prepared_verification.post_patch.model_copy(
                update={"proposal_digest": "f" * 64}
            )
        }

    with pytest.raises(RegressionVerificationError):
        run_verification(prepared_verification, gateway, **updates)

    assert gateway.calls == []


def test_application_metadata_mismatch_blocks_before_execution(
    prepared_verification: PreparedVerification,
) -> None:
    item = prepared_verification.application.files[0]
    changed_file = item.model_copy(update={"candidate_size_bytes": item.candidate_size_bytes + 1})
    application = prepared_verification.application.model_copy(update={"files": (changed_file,)})
    gateway = Gateway(None)

    with pytest.raises(RegressionVerificationError, match="metadata"):
        run_verification(
            prepared_verification,
            gateway,
            application_result=application,
        )

    assert gateway.calls == []


def test_stale_applied_workspace_blocks_before_execution(
    prepared_verification: PreparedVerification,
) -> None:
    prepared_verification.source.write_bytes(b"stale\n")
    gateway = Gateway(None)

    with pytest.raises(RegressionVerificationError, match="applied candidate"):
        run_verification(prepared_verification, gateway)

    assert gateway.calls == []


@pytest.mark.parametrize("mismatch", ["command", "argv"])
def test_post_gateway_identity_mismatch_is_rejected(
    prepared_verification: PreparedVerification,
    mismatch: str,
) -> None:
    command_id = "unit_tests" if mismatch == "command" else "regression_tests"
    argv = (
        ("unexpected",)
        if mismatch == "argv"
        else prepared_verification.task.approved_commands["regression_tests"].argv
    )
    gateway = Gateway(execution(command_id, argv))

    with pytest.raises(RegressionVerificationError, match="inconsistent"):
        run_verification(prepared_verification, gateway)

    assert gateway.calls == ["regression_tests"]


def test_command_time_target_mutation_returns_no_result(
    prepared_verification: PreparedVerification,
) -> None:
    gateway = Gateway(
        execution(
            "regression_tests",
            prepared_verification.task.approved_commands["regression_tests"].argv,
        ),
        mutation=(prepared_verification.source, b"mutated\n"),
    )

    with pytest.raises(RegressionVerificationError, match="modified a proposal target"):
        run_verification(prepared_verification, gateway)

    assert gateway.calls == ["regression_tests"]


def test_operational_error_propagates_without_retry(
    prepared_verification: PreparedVerification,
) -> None:
    error = ApprovedCommandExecutionError("bounded executor failed")
    gateway = Gateway(error=error)

    with pytest.raises(ApprovedCommandExecutionError) as caught:
        run_verification(prepared_verification, gateway)

    assert caught.value is error
    assert gateway.calls == ["regression_tests"]
