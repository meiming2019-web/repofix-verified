"""Tests for the narrow hidden-verification path runner."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import repofix.runners.hidden_verification as runner_module
from repofix.hidden import HiddenVerificationResult
from repofix.tasks import AgentTaskSpec, ApprovedCommand


def test_runner_loads_and_resolves_once_then_executes_one_hidden_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_path = tmp_path / "evaluator/task.yaml"
    task = AgentTaskSpec.model_validate(
        {
            "task_id": "task",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0" * 40,
            "issue_title": "Issue",
            "issue_body": "Body",
            "approved_commands": {"public": {"argv": ["pytest", "-q"]}},
            "allowed_source_paths": ["src"],
            "timeout_seconds": 30,
        }
    )
    hidden_specification = object()
    reproduction = object()
    regression = object()
    command = ApprovedCommand(argv=("pytest", "-q", "/external/test.py"))
    resolution = SimpleNamespace(command_id="hidden", command=command)
    prior = [object() for _ in range(6)]
    load_calls: list[Path] = []
    agent_view_calls = 0
    resolution_calls: list[dict[str, object]] = []
    executor_calls: list[dict[str, object]] = []
    verifier_calls: list[dict[str, object]] = []
    sentinel = object()

    class Bundle:
        def __init__(self) -> None:
            self.reproduction = reproduction
            self.regression = regression
            self.hidden_verification = hidden_specification

        def agent_view(self) -> AgentTaskSpec:
            nonlocal agent_view_calls
            agent_view_calls += 1
            return task

    bundle = Bundle()

    def load(path: Path) -> object:
        load_calls.append(path)
        return bundle

    def resolve(**kwargs: object) -> object:
        resolution_calls.append(kwargs)
        return resolution

    class Executor:
        def __init__(self, **kwargs: object) -> None:
            executor_calls.append(kwargs)

    def verify(**kwargs: object) -> object:
        verifier_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(runner_module, "load_evaluator_task_bundle", load)
    monkeypatch.setattr(runner_module, "resolve_hidden_command", resolve)
    monkeypatch.setattr(runner_module, "LocalApprovedCommandExecutor", Executor)
    monkeypatch.setattr(runner_module, "verify_hidden_behavior", verify)

    result = runner_module.run_hidden_verification_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        original_reproduction_result=prior[0],  # type: ignore[arg-type]
        proposal=prior[1],  # type: ignore[arg-type]
        baseline_result=prior[2],  # type: ignore[arg-type]
        application_result=prior[3],  # type: ignore[arg-type]
        post_patch_reproduction_result=prior[4],  # type: ignore[arg-type]
        regression_verification_result=prior[5],  # type: ignore[arg-type]
    )

    assert result is sentinel
    assert load_calls == [task_path]
    assert agent_view_calls == 1
    assert resolution_calls == [
        {
            "workspace_root": workspace,
            "evaluator_assets_root": task_path.parent,
            "specification": hidden_specification,
        }
    ]
    assert executor_calls == [
        {
            "workspace_root": workspace,
            "approved_commands": {resolution.command_id: resolution.command},
            "timeout_seconds": task.timeout_seconds,
        }
    ]
    assert len(verifier_calls) == 1
    call = verifier_calls[0]
    assert call["task"] is task
    assert call["hidden_specification"] is hidden_specification
    assert call["resolved_hidden_command"] is resolution
    assert isinstance(call["evaluator_command_gateway"], Executor)
    assert "model" not in call
    assert "gold_patch" not in call
    assert "application_gateway" not in call
    assert set(HiddenVerificationResult.model_fields).isdisjoint(
        {"final_verdict", "resolved", "gold_patch"}
    )
