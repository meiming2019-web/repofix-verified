"""Tests for provenance-gated protected workspace inspection."""

from pathlib import Path

import pytest

import repofix.policy.verification as verification_module
from repofix.execution import CommandTerminationReason
from repofix.hidden import (
    HIDDEN_VERIFICATION_FAILED_SUMMARY,
    HIDDEN_VERIFICATION_INCONCLUSIVE_SUMMARY,
    HiddenVerificationResult,
    HiddenVerificationStatus,
)
from repofix.policy import (
    POLICY_VERIFICATION_FAILED_SUMMARY,
    POLICY_VERIFICATION_PASSED_SUMMARY,
    PolicyRuleId,
    PolicyVerificationError,
    PolicyVerificationStatus,
    verify_patch_policy,
)
from repofix.reproduction import ReproductionEvidence
from ..regression.conftest import execution
from .conftest import PreparedPolicy


def run_policy(prepared: PreparedPolicy, **updates: object):
    chain = prepared.hidden_chain
    public = chain.public
    values = {
        "workspace_root": public.workspace,
        "task": public.task,
        "policy_specification": prepared.specification,
        "proposal": public.proposal,
        "application_result": public.application,
        "post_patch_reproduction_result": public.post_patch,
        "regression_verification_result": chain.regression,
        "hidden_verification_result": prepared.hidden_result,
    }
    values.update(updates)
    return verify_patch_policy(**values)  # type: ignore[arg-type]


def test_unchanged_surfaces_pass_with_complete_provenance(
    prepared_policy: PreparedPolicy,
) -> None:
    result = run_policy(prepared_policy)

    assert result.status is PolicyVerificationStatus.POLICY_PASSED
    assert result.findings == ()
    assert result.verification_summary == POLICY_VERIFICATION_PASSED_SUMMARY
    assert result.hidden_verification_status is (
        HiddenVerificationStatus.HIDDEN_COMMAND_PASSED
    )


@pytest.mark.parametrize("mutation", ["changed", "size", "missing", "directory"])
def test_protected_file_final_state_violations_are_findings_not_errors(
    prepared_policy: PreparedPolicy,
    mutation: str,
) -> None:
    path = prepared_policy.protected_file
    if mutation == "changed":
        path.write_bytes(b"protected fixturE\n")
    elif mutation == "size":
        path.write_bytes(b"larger protected fixture\n")
    elif mutation == "missing":
        path.unlink()
    else:
        path.unlink()
        path.mkdir()

    result = run_policy(prepared_policy)

    assert result.status is PolicyVerificationStatus.POLICY_FAILED
    assert [(item.rule_id, item.path) for item in result.findings] == [
        (PolicyRuleId.PROTECTED_FILE_INTEGRITY, "tests/protected.py")
    ]
    assert result.verification_summary == POLICY_VERIFICATION_FAILED_SUMMARY


def test_protected_file_symlink_is_a_finding_without_following_target(
    prepared_policy: PreparedPolicy,
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-external-protected"
    outside.write_bytes(prepared_policy.protected_file.read_bytes())
    prepared_policy.protected_file.unlink()
    try:
        prepared_policy.protected_file.symlink_to(outside)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symbolic links are not supported on this host: {error}")

    result = run_policy(prepared_policy)

    assert result.status is PolicyVerificationStatus.POLICY_FAILED
    assert result.findings[0].rule_id is PolicyRuleId.PROTECTED_FILE_INTEGRITY
    assert str(outside) not in repr(result.model_dump())


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_exact_forbidden_path_presence_is_a_finding(
    prepared_policy: PreparedPolicy,
    kind: str,
    tmp_path: Path,
) -> None:
    forbidden = prepared_policy.hidden_chain.public.workspace / "tests/conftest.py"
    if kind == "file":
        forbidden.write_text("# forbidden\n", encoding="utf-8")
    else:
        outside = tmp_path.parent / f"{tmp_path.name}-external-forbidden"
        outside.write_text("external\n", encoding="utf-8")
        try:
            forbidden.symlink_to(outside)
        except (NotImplementedError, OSError) as error:
            pytest.skip(f"symbolic links are not supported on this host: {error}")

    result = run_policy(prepared_policy)

    assert result.status is PolicyVerificationStatus.POLICY_FAILED
    assert [(item.rule_id, item.path) for item in result.findings] == [
        (PolicyRuleId.FORBIDDEN_PATH_PRESENT, "tests/conftest.py")
    ]


def test_findings_have_stable_rule_and_path_order(prepared_policy: PreparedPolicy) -> None:
    prepared_policy.protected_file.unlink()
    workspace = prepared_policy.hidden_chain.public.workspace
    (workspace / "conftest.py").write_text("# forbidden\n", encoding="utf-8")
    (workspace / "tests/conftest.py").write_text("# forbidden\n", encoding="utf-8")

    result = run_policy(prepared_policy)

    assert [(item.rule_id.value, item.path) for item in result.findings] == sorted(
        (item.rule_id.value, item.path) for item in result.findings
    )


@pytest.mark.parametrize(
    "artifact",
    ["task", "proposal", "application", "post_patch", "regression", "hidden"],
)
def test_stale_or_forged_provenance_rejects_before_filesystem_inspection(
    prepared_policy: PreparedPolicy,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    chain = prepared_policy.hidden_chain
    public = chain.public
    updates: dict[str, object]
    if artifact == "task":
        updates = {"task": public.task.model_copy(update={"issue_body": "changed"})}
    elif artifact == "proposal":
        updates = {
            "application_result": public.application.model_copy(
                update={"proposal_digest": "f" * 64}
            )
        }
    elif artifact == "application":
        updates = {
            "regression_verification_result": chain.regression.model_copy(
                update={"application_result_fingerprint": "f" * 64}
            )
        }
    elif artifact == "post_patch":
        updates = {
            "regression_verification_result": chain.regression.model_copy(
                update={"post_patch_reproduction_fingerprint": "f" * 64}
            )
        }
    elif artifact == "regression":
        updates = {
            "hidden_verification_result": prepared_policy.hidden_result.model_copy(
                update={"regression_verification_fingerprint": "f" * 64}
            )
        }
    else:
        updates = {
            "hidden_verification_result": prepared_policy.hidden_result.model_copy(
                update={"task_fingerprint": "f" * 64}
            )
        }

    def fail_inspection(**kwargs: object) -> object:
        raise AssertionError("filesystem inspection must not start")

    monkeypatch.setattr(verification_module, "capture_applied_targets", fail_inspection)
    monkeypatch.setattr(verification_module, "_inspect_policy", fail_inspection)
    with pytest.raises(PolicyVerificationError):
        run_policy(prepared_policy, **updates)


def _nonpassing_hidden(
    prepared: PreparedPolicy,
    status: HiddenVerificationStatus,
) -> HiddenVerificationResult:
    current = prepared.hidden_result
    inconclusive = status is HiddenVerificationStatus.INCONCLUSIVE
    evidence = ReproductionEvidence.from_execution_result(
        execution(
            current.command_id,
            current.evidence.argv,
            exit_code=None if inconclusive else 1,
            termination=(
                CommandTerminationReason.TIMED_OUT
                if inconclusive
                else CommandTerminationReason.COMPLETED
            ),
        )
    )
    summary = (
        HIDDEN_VERIFICATION_INCONCLUSIVE_SUMMARY
        if inconclusive
        else HIDDEN_VERIFICATION_FAILED_SUMMARY
    )
    return HiddenVerificationResult.model_validate(
        {
            **current.model_dump(),
            "evidence": evidence.model_dump(),
            "status": status,
            "verification_summary": summary,
        }
    )


@pytest.mark.parametrize(
    "status",
    [HiddenVerificationStatus.HIDDEN_COMMAND_FAILED, HiddenVerificationStatus.INCONCLUSIVE],
)
def test_hidden_nonpass_blocks_before_policy_inspection(
    prepared_policy: PreparedPolicy,
    monkeypatch: pytest.MonkeyPatch,
    status: HiddenVerificationStatus,
) -> None:
    monkeypatch.setattr(
        verification_module,
        "_inspect_policy",
        lambda **kwargs: pytest.fail("policy inspection must not run"),
    )
    with pytest.raises(PolicyVerificationError, match="hidden pass gate"):
        run_policy(
            prepared_policy,
            hidden_verification_result=_nonpassing_hidden(prepared_policy, status),
        )


def test_stale_applied_candidate_is_an_error_not_a_policy_finding(
    prepared_policy: PreparedPolicy,
) -> None:
    public = prepared_policy.hidden_chain.public
    public.source.write_bytes(b"stale candidate\n")

    with pytest.raises(PolicyVerificationError, match="applied candidate"):
        run_policy(prepared_policy)


def test_workspace_errors_are_sanitized(prepared_policy: PreparedPolicy, tmp_path: Path) -> None:
    missing = tmp_path / "private/missing"
    with pytest.raises(PolicyVerificationError, match="could not be resolved") as caught:
        run_policy(prepared_policy, workspace_root=missing)
    assert str(tmp_path) not in str(caught.value)
