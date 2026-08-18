"""Tests for pure complete-chain final evaluation."""

import builtins
from contextlib import nullcontext
import inspect
import os
import socket
import subprocess

import pytest

import repofix.final_evaluation.finalization as finalization_module
from repofix.final_evaluation import (
    FINAL_EVALUATION_PASSED_SUMMARY,
    FinalEvaluationError,
    FinalEvaluationStatus,
    finalize_evaluation,
)
from repofix.hidden import (
    HiddenVerificationStatus,
    compute_hidden_verification_fingerprint,
)
from repofix.patching import compute_proposal_digest
from repofix.policy import (
    FORBIDDEN_PATH_PRESENT_SUMMARY,
    POLICY_VERIFICATION_FAILED_SUMMARY,
    PolicyFinding,
    PolicyRuleId,
    PolicyVerificationResult,
    PolicyVerificationStatus,
    compute_policy_verification_fingerprint,
)
from repofix.regression import RegressionBaselineStatus, RegressionVerificationStatus
from repofix.reproduction import PostPatchReproductionStatus
from tests.unit.final_evaluation.conftest import PreparedFinalEvaluation


def run_final(
    prepared: PreparedFinalEvaluation, **updates: object
):
    arguments = prepared.arguments()
    arguments.update(updates)
    return finalize_evaluation(**arguments)  # type: ignore[arg-type]


def test_complete_success_chain_issues_minimal_terminal_certificate(
    prepared_final_evaluation: PreparedFinalEvaluation,
) -> None:
    prepared = prepared_final_evaluation
    public = prepared.policy_chain.hidden_chain.public

    result = run_final(prepared)

    assert result.task_id == public.task.task_id
    assert result.task_fingerprint == public.original.task_fingerprint
    assert result.proposal_digest == public.proposal.proposal_digest
    assert result.policy_verification_fingerprint == (
        compute_policy_verification_fingerprint(prepared.policy_result)
    )
    assert result.status is FinalEvaluationStatus.EVALUATOR_PASSED
    assert result.verification_summary == FINAL_EVALUATION_PASSED_SUMMARY


@pytest.mark.parametrize(
    ("argument", "field"),
    [
        ("original_reproduction_result", "task_fingerprint"),
        ("proposal", "reproduction_expectation_fingerprint"),
        ("regression_baseline_result", "original_reproduction_run_fingerprint"),
        ("application_result", "proposal_digest"),
        ("regression_verification_result", "regression_specification_fingerprint"),
        ("hidden_verification_result", "regression_baseline_fingerprint"),
        ("policy_verification_result", "application_result_fingerprint"),
        ("hidden_verification_result", "post_patch_reproduction_fingerprint"),
        ("policy_verification_result", "regression_verification_fingerprint"),
        ("hidden_verification_result", "hidden_specification_fingerprint"),
        ("policy_verification_result", "hidden_verification_fingerprint"),
        ("policy_verification_result", "policy_specification_fingerprint"),
    ],
)
def test_each_provenance_boundary_rejects_stale_or_forged_fingerprints(
    prepared_final_evaluation: PreparedFinalEvaluation,
    argument: str,
    field: str,
) -> None:
    artifact = prepared_final_evaluation.arguments()[argument]
    forged = artifact.model_copy(update={field: "f" * 64})  # type: ignore[attr-defined]

    with pytest.raises(FinalEvaluationError):
        run_final(prepared_final_evaluation, **{argument: forged})


@pytest.mark.parametrize(
    "argument",
    [
        "original_reproduction_result",
        "proposal",
        "regression_baseline_result",
        "application_result",
        "post_patch_reproduction_result",
        "regression_verification_result",
        "hidden_verification_result",
        "policy_verification_result",
    ],
)
def test_task_id_mismatch_at_every_artifact_boundary_is_rejected(
    prepared_final_evaluation: PreparedFinalEvaluation,
    argument: str,
) -> None:
    artifact = prepared_final_evaluation.arguments()[argument]
    if argument == "original_reproduction_result":
        state = artifact.state.model_copy(update={"task_id": "different-task"})  # type: ignore[attr-defined]
        forged = artifact.model_copy(update={"state": state})  # type: ignore[attr-defined]
    elif argument == "proposal":
        changed = artifact.model_copy(update={"task_id": "different-task"})  # type: ignore[attr-defined]
        forged = changed.model_copy(
            update={
                "proposal_digest": compute_proposal_digest(
                    task_id=changed.task_id,
                    task_fingerprint=changed.task_fingerprint,
                    reproduction_expectation_fingerprint=(
                        changed.reproduction_expectation_fingerprint
                    ),
                    reproduction_run_fingerprint=changed.reproduction_run_fingerprint,
                    hypothesis_id=changed.hypothesis_id,
                    model_summary=changed.model_summary,
                    validation_status=changed.validation_status,
                    validation_summary=changed.validation_summary,
                    edits=changed.edits,
                    file_snapshots=changed.file_snapshots,
                    unified_diff=changed.unified_diff,
                )
            }
        )
    else:
        forged = artifact.model_copy(update={"task_id": "different-task"})  # type: ignore[attr-defined]

    with pytest.raises(FinalEvaluationError, match="task identity"):
        run_final(prepared_final_evaluation, **{argument: forged})


@pytest.mark.parametrize(
    ("argument", "field", "value"),
    [
        ("regression_baseline_result", "command_id", "unit_tests"),
        ("post_patch_reproduction_result", "command_id", "regression_tests"),
        ("hidden_verification_result", "command_id", "different-hidden"),
        (
            "hidden_verification_result",
            "regression_verification_status",
            RegressionVerificationStatus.INCONCLUSIVE,
        ),
        ("policy_verification_result", "application_status", "not-applied"),
        (
            "policy_verification_result",
            "post_patch_reproduction_status",
            PostPatchReproductionStatus.INCONCLUSIVE,
        ),
        (
            "policy_verification_result",
            "regression_verification_status",
            RegressionVerificationStatus.INCONCLUSIVE,
        ),
        (
            "policy_verification_result",
            "hidden_verification_status",
            HiddenVerificationStatus.INCONCLUSIVE,
        ),
    ],
)
def test_inconsistent_commands_and_copied_statuses_are_rejected(
    prepared_final_evaluation: PreparedFinalEvaluation,
    argument: str,
    field: str,
    value: object,
) -> None:
    artifact = prepared_final_evaluation.arguments()[argument]
    forged = artifact.model_copy(update={field: value})  # type: ignore[attr-defined]

    warning = pytest.warns(UserWarning) if value == "not-applied" else nullcontext()
    with warning:
        with pytest.raises(FinalEvaluationError):
            run_final(prepared_final_evaluation, **{argument: forged})


def test_application_file_metadata_must_match_proposal_snapshot(
    prepared_final_evaluation: PreparedFinalEvaluation,
) -> None:
    application = prepared_final_evaluation.arguments()["application_result"]
    file = application.files[0]  # type: ignore[attr-defined]
    forged_file = file.model_copy(update={"candidate_size_bytes": file.candidate_size_bytes + 1})
    forged = application.model_copy(update={"files": (forged_file,)})  # type: ignore[attr-defined]

    with pytest.raises(FinalEvaluationError, match="metadata"):
        run_final(prepared_final_evaluation, application_result=forged)


def test_hidden_launcher_arguments_are_bound_without_resolving_the_asset_path(
    prepared_final_evaluation: PreparedFinalEvaluation,
) -> None:
    hidden = prepared_final_evaluation.arguments()["hidden_verification_result"]
    evidence = hidden.evidence.model_copy(update={"argv": ("forged-launcher",)})  # type: ignore[attr-defined]
    forged_hidden = hidden.model_copy(update={"evidence": evidence})  # type: ignore[attr-defined]
    forged_policy = prepared_final_evaluation.policy_result.model_copy(
        update={
            "hidden_verification_fingerprint": (
                compute_hidden_verification_fingerprint(forged_hidden)
            )
        }
    )

    with pytest.raises(FinalEvaluationError, match="hidden command arguments"):
        run_final(
            prepared_final_evaluation,
            hidden_verification_result=forged_hidden,
            policy_verification_result=forged_policy,
        )


@pytest.mark.parametrize(
    ("argument", "field", "value"),
    [
        ("original_reproduction_result", "attempts", ()),
        ("regression_baseline_result", "status", RegressionBaselineStatus.FAILED),
        ("application_result", "status", "not-applied"),
        (
            "post_patch_reproduction_result",
            "status",
            PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_STILL_REPRODUCED,
        ),
        (
            "regression_verification_result",
            "status",
            RegressionVerificationStatus.REGRESSION_COMMAND_FAILED,
        ),
        (
            "hidden_verification_result",
            "status",
            HiddenVerificationStatus.HIDDEN_COMMAND_FAILED,
        ),
        (
            "hidden_verification_result",
            "status",
            HiddenVerificationStatus.INCONCLUSIVE,
        ),
        (
            "policy_verification_result",
            "status",
            PolicyVerificationStatus.POLICY_FAILED,
        ),
    ],
)
def test_every_unsuccessful_gate_refuses_a_final_success_certificate(
    prepared_final_evaluation: PreparedFinalEvaluation,
    argument: str,
    field: str,
    value: object,
) -> None:
    artifact = prepared_final_evaluation.arguments()[argument]
    forged = artifact.model_copy(update={field: value})  # type: ignore[attr-defined]

    warning = pytest.warns(UserWarning) if value == "not-applied" else nullcontext()
    with warning:
        with pytest.raises(FinalEvaluationError):
            run_final(prepared_final_evaluation, **{argument: forged})


def test_valid_policy_failure_and_nonempty_findings_are_authoritative_upstream(
    prepared_final_evaluation: PreparedFinalEvaluation,
) -> None:
    current = prepared_final_evaluation.policy_result
    finding = PolicyFinding(
        rule_id=PolicyRuleId.FORBIDDEN_PATH_PRESENT,
        path="tests/conftest.py",
        summary=FORBIDDEN_PATH_PRESENT_SUMMARY,
    )
    failed = PolicyVerificationResult.model_validate(
        {
            **current.model_dump(),
            "status": PolicyVerificationStatus.POLICY_FAILED,
            "findings": (finding,),
            "verification_summary": POLICY_VERIFICATION_FAILED_SUMMARY,
        }
    )

    with pytest.raises(FinalEvaluationError, match="policy pass gate"):
        run_final(prepared_final_evaluation, policy_verification_result=failed)


def test_inputs_are_canonicalized_without_mutation_and_errors_are_sanitized(
    prepared_final_evaluation: PreparedFinalEvaluation,
    tmp_path,
) -> None:
    before = {
        name: value.model_dump(mode="json")
        for name, value in prepared_final_evaluation.arguments().items()
    }
    result = run_final(prepared_final_evaluation)
    after = {
        name: value.model_dump(mode="json")
        for name, value in prepared_final_evaluation.arguments().items()
    }
    forged = prepared_final_evaluation.policy_result.model_copy(
        update={"task_fingerprint": "not-a-hash"}
    )

    assert before == after
    assert result.status is FinalEvaluationStatus.EVALUATOR_PASSED
    with pytest.raises(FinalEvaluationError) as caught:
        run_final(prepared_final_evaluation, policy_verification_result=forged)
    assert str(tmp_path) not in str(caught.value)
    with pytest.raises(FinalEvaluationError, match="invalid type"):
        run_final(prepared_final_evaluation, policy_verification_result=object())


def test_finalizer_has_no_workspace_or_side_effect_capabilities(
    prepared_final_evaluation: PreparedFinalEvaluation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "workspace_root" not in inspect.signature(finalize_evaluation).parameters

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("final evaluation attempted a forbidden side effect")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    verify_calls = 0
    original_verify = finalization_module._verify_chain

    def verify_once(**kwargs: object):
        nonlocal verify_calls
        verify_calls += 1
        return original_verify(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(finalization_module, "_verify_chain", verify_once)

    result = run_final(prepared_final_evaluation)

    assert result.status is FinalEvaluationStatus.EVALUATOR_PASSED
    assert verify_calls == 1
    source = inspect.getsource(finalization_module)
    assert "workspace_root" not in source
    assert "LocalApprovedCommandExecutor" not in source
    assert "verify_patch_policy" not in source
    assert "verify_hidden_behavior" not in source
    assert "verify_post_patch_regression" not in source
    assert "verify_post_patch_reproduction" not in source
    assert "apply_validated_patch_proposal" not in source
    assert "subprocess" not in source
