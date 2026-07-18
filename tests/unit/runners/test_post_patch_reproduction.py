"""Tests for post-patch reproduction runner orchestration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import repofix.runners.post_patch_reproduction as runner_module
from repofix.agent.reproduction_loop import compute_task_fingerprint
from repofix.execution import ApprovedCommandExecutionError
from repofix.patching import PatchApplicationStatus
from repofix.reproduction import (
    POST_PATCH_NOT_REPRODUCED_SUMMARY,
    PostPatchReproductionResult,
    PostPatchReproductionStatus,
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionStatus,
    ReproductionTerminationReason,
    ReproductionVerdict,
    compute_reproduction_expectation_fingerprint,
)
from repofix.tasks import AgentTaskSpec


def _inputs(tmp_path: Path):
    task = AgentTaskSpec.model_validate(
        {
            "task_id": "post-patch-runner",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0" * 40,
            "issue_title": "Target failure",
            "issue_body": "The target behavior fails.",
            "approved_commands": {"unit_tests": {"argv": ["pytest", "-q"]}},
            "allowed_source_paths": ["src"],
            "patchable_source_paths": ["src"],
            "timeout_seconds": 30,
        }
    )
    expectation = ReproductionExpectation.model_validate(
        {
            "command_id": "unit_tests",
            "expected_exit_codes": [1],
            "required_fragments": [
                {"fragment_id": "target", "stream": "combined", "text": "TARGET"}
            ],
        }
    )
    reproduction_result = SimpleNamespace(
        state=SimpleNamespace(task_id=task.task_id),
        task_fingerprint=compute_task_fingerprint(task),
        reproduction_expectation_fingerprint=(
            compute_reproduction_expectation_fingerprint(expectation)
        ),
    )
    bundle = SimpleNamespace(agent_view=lambda: task, reproduction=expectation)
    return task, expectation, reproduction_result, bundle, object(), object(), tmp_path


def _immutable_result() -> PostPatchReproductionResult:
    evidence = ReproductionEvidence(
        command_id="unit_tests",
        argv=("pytest", "-q"),
        termination_reason=ReproductionTerminationReason.COMPLETED,
        exit_code=0,
        stdout="1 passed\n",
        stderr="",
        stdout_bytes=9,
        stderr_bytes=0,
        had_decode_errors=False,
    )
    verdict = ReproductionVerdict(
        status=ReproductionStatus.NOT_REPRODUCED,
        command_id="unit_tests",
        exit_code=0,
        reasons=("command completed with exit code zero",),
        matched_required_fragment_ids=(),
        missing_required_fragment_ids=("target",),
        forbidden_fragment_ids_found=(),
    )
    return PostPatchReproductionResult(
        task_id="post-patch-runner",
        task_fingerprint="a" * 64,
        reproduction_expectation_fingerprint="b" * 64,
        original_reproduction_run_fingerprint="c" * 64,
        proposal_digest="d" * 64,
        application_status=PatchApplicationStatus.APPLIED,
        status=PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED,
        command_id="unit_tests",
        evidence=evidence,
        verifier_verdict=verdict,
        verification_summary=POST_PATCH_NOT_REPRODUCED_SUMMARY,
    )


def test_runner_loads_bundle_constructs_gateway_once_and_verifies_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task, _, reproduction_result, bundle, proposal, application, workspace = _inputs(tmp_path)
    task_path = tmp_path / "task.yaml"
    expected = _immutable_result()
    loads: list[Path] = []
    gateway_arguments: list[dict[str, object]] = []
    verification_calls: list[dict[str, object]] = []

    def load(path: Path) -> object:
        loads.append(path)
        return bundle

    class Gateway:
        def __init__(self, **kwargs: object) -> None:
            gateway_arguments.append(kwargs)

    def verify(**kwargs: object) -> PostPatchReproductionResult:
        verification_calls.append(kwargs)
        return expected

    monkeypatch.setattr(runner_module, "load_reproduction_task_bundle", load)
    monkeypatch.setattr(runner_module, "LocalApprovedCommandExecutor", Gateway)
    monkeypatch.setattr(runner_module, "verify_post_patch_reproduction", verify)

    result = runner_module.run_post_patch_reproduction_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        original_reproduction_result=reproduction_result,  # type: ignore[arg-type]
        proposal=proposal,  # type: ignore[arg-type]
        application_result=application,  # type: ignore[arg-type]
    )

    assert result is expected
    assert loads == [task_path]
    assert len(gateway_arguments) == 1
    assert gateway_arguments[0] == {
        "workspace_root": workspace,
        "approved_commands": task.approved_commands,
        "timeout_seconds": task.timeout_seconds,
    }
    assert len(verification_calls) == 1
    assert "model" not in verification_calls[0]
    assert "apply_validated_patch_proposal" not in verification_calls[0]


def test_runner_rejects_current_expectation_mismatch_before_gateway(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, reproduction_result, bundle, proposal, application, workspace = _inputs(tmp_path)
    monkeypatch.setattr(runner_module, "load_reproduction_task_bundle", lambda path: bundle)
    monkeypatch.setattr(
        runner_module,
        "compute_reproduction_expectation_fingerprint",
        lambda expectation: "f" * 64,
    )
    gateway_calls = 0

    class Gateway:
        def __init__(self, **kwargs: object) -> None:
            nonlocal gateway_calls
            gateway_calls += 1

    monkeypatch.setattr(runner_module, "LocalApprovedCommandExecutor", Gateway)

    with pytest.raises(ValueError, match="expectation fingerprint"):
        runner_module.run_post_patch_reproduction_from_paths(
            task_path=tmp_path / "task.yaml",
            workspace_root=workspace,
            original_reproduction_result=reproduction_result,  # type: ignore[arg-type]
            proposal=proposal,  # type: ignore[arg-type]
            application_result=application,  # type: ignore[arg-type]
        )

    assert gateway_calls == 0


def test_verification_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, reproduction_result, bundle, proposal, application, workspace = _inputs(tmp_path)
    monkeypatch.setattr(runner_module, "load_reproduction_task_bundle", lambda path: bundle)
    monkeypatch.setattr(runner_module, "LocalApprovedCommandExecutor", lambda **kwargs: object())
    error = ApprovedCommandExecutionError("execution failed")
    calls = 0

    def fail(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(runner_module, "verify_post_patch_reproduction", fail)

    with pytest.raises(ApprovedCommandExecutionError) as caught:
        runner_module.run_post_patch_reproduction_from_paths(
            task_path=tmp_path / "task.yaml",
            workspace_root=workspace,
            original_reproduction_result=reproduction_result,  # type: ignore[arg-type]
            proposal=proposal,  # type: ignore[arg-type]
            application_result=application,  # type: ignore[arg-type]
        )

    assert caught.value is error
    assert calls == 1
