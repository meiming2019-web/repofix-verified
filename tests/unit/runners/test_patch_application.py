"""Tests for path-based controlled patch application orchestration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import repofix.runners.patch_application as runner_module
from repofix.agent.reproduction_loop import compute_task_fingerprint
from repofix.patching import PatchApplicationError
from repofix.tasks import AgentTaskSpec


class Bundle:
    def __init__(self, task: object) -> None:
        self.task = task
        self.reproduction = object()

    def agent_view(self) -> object:
        return self.task


def _inputs(tmp_path: Path):
    task = AgentTaskSpec.model_validate(
        {
            "task_id": "patch-task",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0" * 40,
            "issue_title": "Wrong return",
            "issue_body": "Wrong value.",
            "approved_commands": {"tests": {"argv": ["pytest"]}},
            "allowed_source_paths": ["src"],
            "patchable_source_paths": ["src"],
            "timeout_seconds": 30,
        }
    )
    result = SimpleNamespace(
        state=SimpleNamespace(task_id=task.task_id),
        task_fingerprint=compute_task_fingerprint(task),
        reproduction_expectation_fingerprint="e" * 64,
    )
    return task, result, object(), tmp_path


def _configure_bundle(
    monkeypatch: pytest.MonkeyPatch,
    patch_task,
    reproduced_result,
    *,
    load_calls: list[Path] | None = None,
) -> None:
    def load(path: Path) -> Bundle:
        if load_calls is not None:
            load_calls.append(path)
        return Bundle(patch_task)

    monkeypatch.setattr(runner_module, "load_reproduction_task_bundle", load)
    monkeypatch.setattr(
        runner_module,
        "compute_reproduction_expectation_fingerprint",
        lambda expectation: reproduced_result.reproduction_expectation_fingerprint,
    )


def test_runner_loads_bundle_and_calls_application_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_task, reproduced_result, proposal, patch_workspace = _inputs(tmp_path)
    expected = object()
    task_path = tmp_path / "task.yaml"
    load_calls: list[Path] = []
    application_calls: list[dict[str, object]] = []
    _configure_bundle(
        monkeypatch,
        patch_task,
        reproduced_result,
        load_calls=load_calls,
    )

    def apply(**kwargs: object) -> object:
        application_calls.append(kwargs)
        return expected

    monkeypatch.setattr(runner_module, "apply_validated_patch_proposal", apply)

    result = runner_module.run_patch_application_from_paths(
        task_path=task_path,
        workspace_root=patch_workspace,
        reproduction_result=reproduced_result,  # type: ignore[arg-type]
        proposal=proposal,  # type: ignore[arg-type]
    )

    assert result is expected
    assert load_calls == [task_path]
    assert len(application_calls) == 1
    assert application_calls[0]["task"] == patch_task
    assert "model" not in application_calls[0]


@pytest.mark.parametrize("mismatch", ["task", "expectation"])
def test_runner_rejects_bundle_mismatch_before_application(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mismatch: str,
) -> None:
    patch_task, reproduced_result, proposal, patch_workspace = _inputs(tmp_path)
    task = (
        patch_task.model_copy(update={"issue_body": "different task"})
        if mismatch == "task"
        else patch_task
    )
    _configure_bundle(monkeypatch, task, reproduced_result)
    if mismatch == "expectation":
        monkeypatch.setattr(
            runner_module,
            "compute_reproduction_expectation_fingerprint",
            lambda expectation: "f" * 64,
        )
    calls = 0

    def fail_if_called(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("application must not be called")

    monkeypatch.setattr(runner_module, "apply_validated_patch_proposal", fail_if_called)

    with pytest.raises(ValueError, match=f"{mismatch} fingerprint"):
        runner_module.run_patch_application_from_paths(
            task_path=patch_workspace / "task.yaml",
            workspace_root=patch_workspace,
            reproduction_result=reproduced_result,  # type: ignore[arg-type]
            proposal=proposal,  # type: ignore[arg-type]
        )

    assert calls == 0


def test_application_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    patch_task, reproduced_result, proposal, patch_workspace = _inputs(tmp_path)
    _configure_bundle(monkeypatch, patch_task, reproduced_result)
    calls = 0

    def fail(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise PatchApplicationError("application failed")

    monkeypatch.setattr(runner_module, "apply_validated_patch_proposal", fail)

    with pytest.raises(PatchApplicationError, match="application failed"):
        runner_module.run_patch_application_from_paths(
            task_path=patch_workspace / "task.yaml",
            workspace_root=patch_workspace,
            reproduction_result=reproduced_result,  # type: ignore[arg-type]
            proposal=proposal,  # type: ignore[arg-type]
        )

    assert calls == 1
