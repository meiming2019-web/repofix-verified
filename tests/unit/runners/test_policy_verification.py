"""Tests for the command-free protected workspace policy runner."""

from pathlib import Path

import pytest

import repofix.runners.policy_verification as runner_module
from repofix.policy import PolicyVerificationResult
from repofix.tasks import AgentTaskSpec


def test_runner_loads_once_uses_agent_view_and_invokes_core_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "evaluator/task.yaml"
    workspace = tmp_path / "workspace"
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
    policy = object()
    prior = [object() for _ in range(5)]
    load_calls: list[Path] = []
    agent_view_calls = 0
    verify_calls: list[dict[str, object]] = []
    sentinel = object()

    class Bundle:
        patch_policy = policy

        def agent_view(self) -> AgentTaskSpec:
            nonlocal agent_view_calls
            agent_view_calls += 1
            return task

    def load(path: Path) -> Bundle:
        load_calls.append(path)
        return Bundle()

    def verify(**kwargs: object) -> object:
        verify_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(runner_module, "load_evaluator_task_bundle", load)
    monkeypatch.setattr(runner_module, "verify_patch_policy", verify)

    result = runner_module.run_policy_verification_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        proposal=prior[0],  # type: ignore[arg-type]
        application_result=prior[1],  # type: ignore[arg-type]
        post_patch_reproduction_result=prior[2],  # type: ignore[arg-type]
        regression_verification_result=prior[3],  # type: ignore[arg-type]
        hidden_verification_result=prior[4],  # type: ignore[arg-type]
    )

    assert result is sentinel
    assert load_calls == [task_path]
    assert agent_view_calls == 1
    assert len(verify_calls) == 1
    call = verify_calls[0]
    assert call["task"] is task
    assert call["policy_specification"] is policy
    assert "command_gateway" not in call
    assert "model" not in call
    assert "gold_patch" not in call
    assert "application_gateway" not in call
    assert set(PolicyVerificationResult.model_fields).isdisjoint(
        {"final_verdict", "resolved", "gold_patch", "command_id", "evidence"}
    )
