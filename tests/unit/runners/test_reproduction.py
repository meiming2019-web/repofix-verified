"""Tests for provider-independent reproduction runner construction."""

from pathlib import Path

import pytest

import repofix.runners.reproduction as reproduction_runner
from repofix.agent import (
    AgentState,
    AgentWorkflow,
    ReproductionAgentRunResult,
)
from repofix.agent.reproduction_loop import compute_task_fingerprint
from repofix.reproduction import compute_reproduction_expectation_fingerprint
from repofix.runners import run_reproduction_from_paths
from repofix.tasks import AgentTaskSpec, TaskSpecLoadError


REPRODUCTION_YAML = """\
task:
  task_id: runner-reproduction
  repository_url: https://github.com/example/project.git
  pre_fix_commit: 0123456789abcdef0123456789abcdef01234567
  issue_title: Target behavior fails
  issue_body: The target behavior produces an incorrect result.
  approved_commands:
    unit_tests:
      argv: [pytest, -q]
  allowed_source_paths: [src, tests]
  timeout_seconds: 300
reproduction:
  command_id: unit_tests
  expected_exit_codes: [1]
  required_fragments:
    - fragment_id: target
      stream: combined
      text: TARGET FAILURE
"""


class UnusedModel:
    def next_action(self, *, task: AgentTaskSpec, state: AgentState):
        raise AssertionError("unit runner should replace the loop")


def test_runner_loads_bundle_and_builds_both_real_gateway_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_path = tmp_path / "reproduction.yaml"
    task_path.write_text(REPRODUCTION_YAML, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured: dict[str, object] = {}

    class FakeTools:
        def __init__(self, **kwargs: object) -> None:
            captured["tool_arguments"] = kwargs

    class FakeCommandGateway:
        def __init__(self, **kwargs: object) -> None:
            captured["command_arguments"] = kwargs

    def fake_loop(**kwargs: object) -> ReproductionAgentRunResult:
        captured["loop_arguments"] = kwargs
        task = kwargs["task"]
        assert isinstance(task, AgentTaskSpec)
        return ReproductionAgentRunResult(
            state=AgentState.initial(
                task.task_id,
                workflow=AgentWorkflow.REPRODUCTION,
                reproduction_command_id="unit_tests",
            ),
            attempts=(),
            task_fingerprint=compute_task_fingerprint(task),
            reproduction_expectation_fingerprint=(
                compute_reproduction_expectation_fingerprint(kwargs["expectation"])
            ),
        )

    monkeypatch.setattr(reproduction_runner, "LocalReadOnlyToolGateway", FakeTools)
    monkeypatch.setattr(
        reproduction_runner, "LocalApprovedCommandExecutor", FakeCommandGateway
    )
    monkeypatch.setattr(reproduction_runner, "run_reproduction_agent_loop", fake_loop)

    result = run_reproduction_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        model=UnusedModel(),
        max_steps=8,
    )

    assert result.state.workflow is AgentWorkflow.REPRODUCTION
    assert captured["tool_arguments"] == {
        "workspace_root": workspace,
        "allowed_source_paths": ("src", "tests"),
    }
    command_arguments = captured["command_arguments"]
    assert isinstance(command_arguments, dict)
    assert command_arguments["workspace_root"] == workspace
    assert command_arguments["timeout_seconds"] == 300
    loop_arguments = captured["loop_arguments"]
    assert isinstance(loop_arguments, dict)
    assert loop_arguments["max_steps"] == 8
    assert "max_reproduction_attempts" not in loop_arguments


def test_runner_multi_attempt_api_is_absent(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="max_reproduction_attempts"):
        run_reproduction_from_paths(  # type: ignore[call-arg]
            task_path=tmp_path / "unused.yaml",
            workspace_root=tmp_path,
            model=UnusedModel(),
            max_steps=8,
            max_reproduction_attempts=2,
        )


@pytest.mark.parametrize("max_steps", [True, 0, -1, 1.5, "8", 21])
def test_runner_rejects_invalid_step_limits(tmp_path: Path, max_steps: object) -> None:
    with pytest.raises(ValueError, match="strict integer from 1 through 20"):
        run_reproduction_from_paths(
            task_path=tmp_path / "unused.yaml",
            workspace_root=tmp_path,
            model=UnusedModel(),
            max_steps=max_steps,  # type: ignore[arg-type]
        )


def test_runner_rejects_agent_only_yaml(tmp_path: Path) -> None:
    task_path = tmp_path / "agent-only.yaml"
    task_path.write_text(
        REPRODUCTION_YAML.split("reproduction:\n", 1)[0],
        encoding="utf-8",
    )

    with pytest.raises(TaskSpecLoadError, match="model validation"):
        run_reproduction_from_paths(
            task_path=task_path,
            workspace_root=tmp_path,
            model=UnusedModel(),
            max_steps=8,
        )
