"""Tests for narrow model-free regression runner orchestration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import repofix.runners.regression_baseline as baseline_runner
import repofix.runners.regression_verification as verification_runner
from repofix.execution import ApprovedCommandExecutionError, LocalExecutionContext
from repofix.regression import RegressionSpecification
from repofix.tasks import AgentTaskSpec


def _bundle() -> object:
    task = AgentTaskSpec.model_validate(
        {
            "task_id": "runner-regression",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0" * 40,
            "issue_title": "failure",
            "issue_body": "target failure",
            "approved_commands": {
                "unit_tests": {"argv": ["pytest", "target"]},
                "regression_tests": {"argv": ["pytest", "regression"]},
            },
            "allowed_source_paths": ["src"],
            "patchable_source_paths": ["src"],
            "timeout_seconds": 30,
        }
    )
    return SimpleNamespace(
        agent_view=lambda: task,
        reproduction=object(),
        regression=RegressionSpecification(command_id="regression_tests"),
    )


@pytest.mark.parametrize(
    ("module", "runner_name", "function_name", "extra"),
    [
        (
            baseline_runner,
            "run_regression_baseline_from_paths",
            "establish_regression_baseline",
            {},
        ),
        (
            verification_runner,
            "run_regression_verification_from_paths",
            "verify_post_patch_regression",
            {
                "baseline_result": object(),
                "application_result": object(),
                "post_patch_reproduction_result": object(),
            },
        ),
    ],
)
def test_runner_loads_once_constructs_one_gateway_and_calls_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: object,
    runner_name: str,
    function_name: str,
    extra: dict[str, object],
) -> None:
    bundle = _bundle()
    expected = object()
    loads: list[Path] = []
    gateway_arguments: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    def load(path: Path) -> object:
        loads.append(path)
        return bundle

    class Gateway:
        def __init__(self, **kwargs: object) -> None:
            gateway_arguments.append(kwargs)

    def invoke(**kwargs: object) -> object:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(module, "load_evaluator_task_bundle", load)
    monkeypatch.setattr(module, "LocalApprovedCommandExecutor", Gateway)
    monkeypatch.setattr(module, function_name, invoke)
    task_path = tmp_path / "task.yaml"
    workspace = tmp_path / "workspace"
    arguments = {
        "task_path": task_path,
        "workspace_root": workspace,
        "original_reproduction_result": object(),
        "proposal": object(),
        **extra,
    }

    result = getattr(module, runner_name)(**arguments)

    assert result is expected
    assert loads == [task_path]
    assert len(gateway_arguments) == 1
    assert "execution_context" not in gateway_arguments[0]
    assert len(calls) == 1
    assert "model" not in calls[0]
    assert "hidden_tests" not in calls[0]
    assert "apply_validated_patch_proposal" not in calls[0]

    context = LocalExecutionContext(trusted_executable_dirs=(tmp_path / "toolchain",))
    loads.clear()
    gateway_arguments.clear()
    calls.clear()
    arguments["execution_context"] = context
    context_result = getattr(module, runner_name)(**arguments)

    assert context_result is expected
    assert gateway_arguments[0]["execution_context"] is context
    assert "execution_context" not in calls[0]


@pytest.mark.parametrize(
    ("module", "runner_name", "function_name", "extra"),
    [
        (
            baseline_runner,
            "run_regression_baseline_from_paths",
            "establish_regression_baseline",
            {},
        ),
        (
            verification_runner,
            "run_regression_verification_from_paths",
            "verify_post_patch_regression",
            {
                "baseline_result": object(),
                "application_result": object(),
                "post_patch_reproduction_result": object(),
            },
        ),
    ],
)
def test_runner_does_not_retry_operational_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: object,
    runner_name: str,
    function_name: str,
    extra: dict[str, object],
) -> None:
    monkeypatch.setattr(module, "load_evaluator_task_bundle", lambda path: _bundle())
    monkeypatch.setattr(module, "LocalApprovedCommandExecutor", lambda **kwargs: object())
    error = ApprovedCommandExecutionError("execution failed")
    calls = 0

    def fail(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(module, function_name, fail)

    with pytest.raises(ApprovedCommandExecutionError) as caught:
        getattr(module, runner_name)(
            task_path=tmp_path / "task.yaml",
            workspace_root=tmp_path,
            original_reproduction_result=object(),
            proposal=object(),
            **extra,
        )

    assert caught.value is error
    assert calls == 1
