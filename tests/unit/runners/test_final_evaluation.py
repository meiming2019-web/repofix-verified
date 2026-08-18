"""Tests for the pure final-evaluation path runner."""

import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

import repofix.runners.final_evaluation as runner_module
from repofix.final_evaluation import (
    FINAL_EVALUATION_PASSED_SUMMARY,
    FinalEvaluationResult,
    FinalEvaluationStatus,
)
from repofix.tasks import AgentTaskSpec


def test_runner_loads_and_finalizes_exactly_once_without_gold_or_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "evaluator/task.yaml"
    task = AgentTaskSpec.model_validate(
        {
            "task_id": "task",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0" * 40,
            "issue_title": "Issue",
            "issue_body": "Body",
            "approved_commands": {
                "reproduction": {"argv": ["pytest", "target"]},
                "regression": {"argv": ["pytest", "regression"]},
            },
            "allowed_source_paths": ["src"],
            "timeout_seconds": 30,
        }
    )
    expectation = object()
    regression_specification = object()
    hidden_specification = object()
    policy_specification = object()
    artifacts = [object() for _ in range(8)]
    sentinel = FinalEvaluationResult(
        task_id="task",
        task_fingerprint="a" * 64,
        proposal_digest="b" * 64,
        policy_verification_fingerprint="c" * 64,
        status=FinalEvaluationStatus.EVALUATOR_PASSED,
        verification_summary=FINAL_EVALUATION_PASSED_SUMMARY,
    )
    load_calls: list[Path] = []
    agent_view_calls = 0
    finalizer_calls: list[dict[str, object]] = []

    class Bundle:
        reproduction = expectation
        regression = regression_specification
        hidden_verification = hidden_specification
        patch_policy = policy_specification

        @property
        def gold_patch(self) -> object:
            raise AssertionError("final evaluation must not access gold patch data")

        def agent_view(self) -> AgentTaskSpec:
            nonlocal agent_view_calls
            agent_view_calls += 1
            return task

    def load(path: Path) -> Bundle:
        load_calls.append(path)
        return Bundle()

    def finalize(**kwargs: object) -> object:
        finalizer_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(runner_module, "load_evaluator_task_bundle", load)
    monkeypatch.setattr(runner_module, "finalize_evaluation", finalize)

    result = runner_module.run_final_evaluation_from_paths(
        task_path=task_path,
        original_reproduction_result=artifacts[0],  # type: ignore[arg-type]
        proposal=artifacts[1],  # type: ignore[arg-type]
        baseline_result=artifacts[2],  # type: ignore[arg-type]
        application_result=artifacts[3],  # type: ignore[arg-type]
        post_patch_reproduction_result=artifacts[4],  # type: ignore[arg-type]
        regression_verification_result=artifacts[5],  # type: ignore[arg-type]
        hidden_verification_result=artifacts[6],  # type: ignore[arg-type]
        policy_verification_result=artifacts[7],  # type: ignore[arg-type]
    )

    assert result is sentinel
    with pytest.raises(ValidationError):
        result.task_id = "changed"  # type: ignore[misc]
    assert load_calls == [task_path]
    assert agent_view_calls == 1
    assert len(finalizer_calls) == 1
    call = finalizer_calls[0]
    assert call["task"] is task
    assert call["reproduction_expectation"] is expectation
    assert call["regression_specification"] is regression_specification
    assert call["hidden_specification"] is hidden_specification
    assert call["policy_specification"] is policy_specification
    assert list(call.values())[5:] == artifacts
    assert "workspace_root" not in inspect.signature(
        runner_module.run_final_evaluation_from_paths
    ).parameters
    assert set(call).isdisjoint(
        {
            "workspace_root",
            "gold_patch",
            "model",
            "command_gateway",
            "executor",
            "application_gateway",
        }
    )
    assert set(FinalEvaluationResult.model_fields).isdisjoint(
        {"gold_patch", "workspace_root", "resolved", "evidence"}
    )
