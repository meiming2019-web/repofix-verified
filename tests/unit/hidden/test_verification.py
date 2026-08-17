"""Tests for gated exactly-once evaluator-only hidden verification."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from repofix.execution import (
    ApprovedCommandExecutionError,
    CommandTerminationReason,
)
from repofix.hidden import (
    HIDDEN_VERIFICATION_FAILED_SUMMARY,
    HIDDEN_VERIFICATION_INCONCLUSIVE_SUMMARY,
    HIDDEN_VERIFICATION_PASSED_SUMMARY,
    HiddenVerificationError,
    HiddenVerificationResult,
    HiddenVerificationStatus,
    verify_hidden_behavior,
)
from repofix.regression import (
    REGRESSION_BASELINE_FAILED_SUMMARY,
    REGRESSION_VERIFICATION_FAILED_SUMMARY,
    REGRESSION_VERIFICATION_INCONCLUSIVE_SUMMARY,
    RegressionBaselineStatus,
    RegressionVerificationStatus,
    compute_regression_baseline_fingerprint,
)
from repofix.reproduction import ReproductionEvidence
from ..regression.conftest import Gateway, execution
from .conftest import PreparedHidden


def run_hidden(
    prepared: PreparedHidden,
    gateway: Gateway,
    **updates: object,
) -> HiddenVerificationResult:
    public = prepared.public
    values = {
        "workspace_root": public.workspace,
        "task": public.task,
        "reproduction_expectation": public.expectation,
        "regression_specification": public.specification,
        "hidden_specification": prepared.specification,
        "original_reproduction_result": public.original,
        "proposal": public.proposal,
        "regression_baseline_result": public.baseline,
        "application_result": public.application,
        "post_patch_reproduction_result": public.post_patch,
        "regression_verification_result": prepared.regression,
        "resolved_hidden_command": prepared.resolution,
        "evaluator_command_gateway": gateway,
    }
    values.update(updates)
    return verify_hidden_behavior(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("exit_code", "termination", "decode_errors", "status", "summary"),
    [
        (
            0,
            CommandTerminationReason.COMPLETED,
            False,
            HiddenVerificationStatus.HIDDEN_COMMAND_PASSED,
            HIDDEN_VERIFICATION_PASSED_SUMMARY,
        ),
        (
            1,
            CommandTerminationReason.COMPLETED,
            False,
            HiddenVerificationStatus.HIDDEN_COMMAND_FAILED,
            HIDDEN_VERIFICATION_FAILED_SUMMARY,
        ),
        (
            None,
            CommandTerminationReason.TIMED_OUT,
            False,
            HiddenVerificationStatus.INCONCLUSIVE,
            HIDDEN_VERIFICATION_INCONCLUSIVE_SUMMARY,
        ),
        (
            None,
            CommandTerminationReason.OUTPUT_LIMIT,
            False,
            HiddenVerificationStatus.INCONCLUSIVE,
            HIDDEN_VERIFICATION_INCONCLUSIVE_SUMMARY,
        ),
        (
            0,
            CommandTerminationReason.COMPLETED,
            True,
            HiddenVerificationStatus.INCONCLUSIVE,
            HIDDEN_VERIFICATION_INCONCLUSIVE_SUMMARY,
        ),
    ],
)
def test_hidden_command_is_classified_once_with_fixed_conservative_summary(
    prepared_hidden: PreparedHidden,
    exit_code: int | None,
    termination: CommandTerminationReason,
    decode_errors: bool,
    status: HiddenVerificationStatus,
    summary: str,
) -> None:
    resolution = prepared_hidden.resolution
    gateway = Gateway(
        execution(
            resolution.command_id,
            resolution.command.argv,
            exit_code=exit_code,
            termination=termination,
            decode_errors=decode_errors,
        )
    )

    result = run_hidden(prepared_hidden, gateway)

    assert result.status is status
    assert result.verification_summary == summary
    assert gateway.calls == [resolution.command_id]
    assert result.regression_verification_status is (
        RegressionVerificationStatus.REGRESSION_COMMAND_PASSED
    )
    assert "repair succeeded" not in result.verification_summary.lower()


@pytest.mark.parametrize(
    "artifact",
    [
        "task",
        "expectation",
        "original",
        "proposal",
        "baseline",
        "application",
        "post_patch",
        "regression",
    ],
)
def test_every_stale_provenance_layer_blocks_before_hidden_execution(
    prepared_hidden: PreparedHidden,
    artifact: str,
) -> None:
    public = prepared_hidden.public
    updates: dict[str, object]
    if artifact == "task":
        updates = {"task": public.task.model_copy(update={"issue_body": "changed"})}
    elif artifact == "expectation":
        updates = {
            "reproduction_expectation": public.expectation.model_copy(
                update={"command_id": "regression_tests"}
            )
        }
    elif artifact == "original":
        updates = {
            "original_reproduction_result": public.original.model_copy(
                update={"task_fingerprint": "f" * 64}
            )
        }
    elif artifact == "proposal":
        updates = {
            "proposal": public.proposal.model_copy(
                update={"task_fingerprint": "f" * 64}
            )
        }
    elif artifact == "baseline":
        updates = {
            "regression_baseline_result": public.baseline.model_copy(
                update={"proposal_digest": "f" * 64}
            )
        }
    elif artifact == "application":
        updates = {
            "application_result": public.application.model_copy(
                update={"reproduction_run_fingerprint": "f" * 64}
            )
        }
    elif artifact == "post_patch":
        updates = {
            "post_patch_reproduction_result": public.post_patch.model_copy(
                update={"proposal_digest": "f" * 64}
            )
        }
    else:
        updates = {
            "regression_verification_result": prepared_hidden.regression.model_copy(
                update={"regression_baseline_fingerprint": "f" * 64}
            )
        }
    gateway = Gateway(None)

    with pytest.raises(HiddenVerificationError):
        run_hidden(prepared_hidden, gateway, **updates)

    assert gateway.calls == []


def _nonpassing_regression(
    prepared: PreparedHidden,
    *,
    inconclusive: bool,
):
    current = prepared.regression
    termination = (
        CommandTerminationReason.TIMED_OUT
        if inconclusive
        else CommandTerminationReason.COMPLETED
    )
    evidence = ReproductionEvidence.from_execution_result(
        execution(
            current.command_id,
            current.evidence.argv,
            exit_code=None if inconclusive else 1,
            termination=termination,
        )
    )
    status = (
        RegressionVerificationStatus.INCONCLUSIVE
        if inconclusive
        else RegressionVerificationStatus.REGRESSION_COMMAND_FAILED
    )
    summary = (
        REGRESSION_VERIFICATION_INCONCLUSIVE_SUMMARY
        if inconclusive
        else REGRESSION_VERIFICATION_FAILED_SUMMARY
    )
    return type(current).model_validate(
        {
            **current.model_dump(),
            "evidence": evidence.model_dump(),
            "status": status,
            "verification_summary": summary,
        }
    )


@pytest.mark.parametrize("inconclusive", [False, True])
def test_public_regression_gate_blocks_before_execution(
    prepared_hidden: PreparedHidden,
    inconclusive: bool,
) -> None:
    gateway = Gateway(None)

    with pytest.raises(HiddenVerificationError, match="public regression pass gate"):
        run_hidden(
            prepared_hidden,
            gateway,
            regression_verification_result=_nonpassing_regression(
                prepared_hidden,
                inconclusive=inconclusive,
            ),
        )

    assert gateway.calls == []


def test_nonpassing_baseline_and_nonapplied_status_block_before_execution(
    prepared_hidden: PreparedHidden,
) -> None:
    public = prepared_hidden.public
    failed_evidence = ReproductionEvidence.from_execution_result(
        execution(
            public.baseline.command_id,
            public.baseline.evidence.argv,
            exit_code=1,
        )
    )
    failed_baseline = type(public.baseline).model_validate(
        {
            **public.baseline.model_dump(),
            "evidence": failed_evidence.model_dump(),
            "status": RegressionBaselineStatus.FAILED,
            "baseline_summary": REGRESSION_BASELINE_FAILED_SUMMARY,
        }
    )
    matching_regression = prepared_hidden.regression.model_copy(
        update={
            "regression_baseline_fingerprint": compute_regression_baseline_fingerprint(
                failed_baseline
            )
        }
    )
    gateway = Gateway(None)
    with pytest.raises(HiddenVerificationError, match="passing regression baseline"):
        run_hidden(
            prepared_hidden,
            gateway,
            regression_baseline_result=failed_baseline,
            regression_verification_result=matching_regression,
        )
    assert gateway.calls == []

    invalid_application = public.application.model_copy(update={"status": "not_applied"})
    with pytest.warns(UserWarning), pytest.raises(HiddenVerificationError):
        run_hidden(
            prepared_hidden,
            gateway,
            application_result=invalid_application,
        )
    assert gateway.calls == []


@pytest.mark.parametrize("identity", ["command_id", "argv"])
def test_gateway_identity_mismatch_is_rejected(
    prepared_hidden: PreparedHidden,
    identity: str,
) -> None:
    resolution = prepared_hidden.resolution
    result = execution(resolution.command_id, resolution.command.argv)
    if identity == "command_id":
        result = result.model_copy(update={"command_id": "other_hidden"})
    else:
        result = result.model_copy(update={"argv": ("pytest", "wrong")})
    gateway = Gateway(result)

    with pytest.raises(HiddenVerificationError, match="inconsistent"):
        run_hidden(prepared_hidden, gateway)

    assert gateway.calls == [resolution.command_id]


@pytest.mark.parametrize("target", ["proposal", "hidden"])
def test_command_time_file_mutation_invalidates_hidden_result(
    prepared_hidden: PreparedHidden,
    target: str,
) -> None:
    resolution = prepared_hidden.resolution
    mutation_path: Path
    if target == "proposal":
        mutation_path = prepared_hidden.public.source
    else:
        mutation_path = prepared_hidden.hidden_file
    gateway = Gateway(
        execution(resolution.command_id, resolution.command.argv),
        mutation=(mutation_path, b"mutated\n"),
    )

    with pytest.raises(HiddenVerificationError, match="modified"):
        run_hidden(prepared_hidden, gateway)

    assert gateway.calls == [resolution.command_id]


def test_stale_applied_workspace_is_rejected_before_execution(
    prepared_hidden: PreparedHidden,
) -> None:
    prepared_hidden.public.source.write_bytes(b"stale\n")
    gateway = Gateway(None)

    with pytest.raises(HiddenVerificationError, match="applied candidate"):
        run_hidden(prepared_hidden, gateway)

    assert gateway.calls == []


def test_operational_executor_error_propagates_without_retry(
    prepared_hidden: PreparedHidden,
) -> None:
    error = ApprovedCommandExecutionError("bounded executor failed")
    gateway = Gateway(error=error)

    with pytest.raises(ApprovedCommandExecutionError) as caught:
        run_hidden(prepared_hidden, gateway)

    assert caught.value is error
    assert gateway.calls == [prepared_hidden.resolution.command_id]


def test_hidden_result_is_frozen_and_rejects_unknown_fields(
    prepared_hidden: PreparedHidden,
) -> None:
    resolution = prepared_hidden.resolution
    result = run_hidden(
        prepared_hidden,
        Gateway(execution(resolution.command_id, resolution.command.argv)),
    )

    with pytest.raises(ValidationError):
        result.status = HiddenVerificationStatus.INCONCLUSIVE  # type: ignore[misc]
    with pytest.raises(ValidationError):
        HiddenVerificationResult.model_validate(
            {**result.model_dump(), "final_repair_verdict": True}
        )
