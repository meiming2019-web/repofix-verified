"""Tests for the RepoFix command-line interface."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import repofix.cli as cli_module
from repofix.agent import AgentPhase, AgentProtocolError, AgentState, ToolExecutionError
from repofix.models import ModelExecutionError
from repofix.tasks import TaskSpecLoadError


runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(cli_module.app, ["version"])

    assert result.exit_code == 0
    assert "RepoFix Verified 0.1.0" in result.stdout


def cli_paths(tmp_path: Path) -> tuple[Path, Path]:
    task = tmp_path / "task.yaml"
    workspace = tmp_path / "workspace"
    task.write_text("placeholder", encoding="utf-8")
    workspace.mkdir()
    return task, workspace


def arguments(task: Path, workspace: Path, *extra: str) -> list[str]:
    return [
        "investigate",
        "--task",
        str(task),
        "--workspace",
        str(workspace),
        "--model",
        "configured-model",
        *extra,
    ]


def finished_state() -> AgentState:
    return AgentState(
        task_id="cli-task",
        phase=AgentPhase.FINISHED,
        issue_understanding=None,
        hypotheses=(),
        observations=(),
        step_count=5,
        terminal_summary="Read-only investigation complete.",
        failure_reason=None,
    )


def failed_state() -> AgentState:
    return AgentState(
        task_id="cli-task",
        phase=AgentPhase.FAILED,
        issue_understanding=None,
        hypotheses=(),
        observations=(),
        step_count=8,
        terminal_summary=None,
        failure_reason="investigation exceeded the 8-step budget",
    )


def install_fake_model(monkeypatch: pytest.MonkeyPatch) -> object:
    fake_model = object()

    def create_model(*, model: str) -> object:
        assert model == "configured-model"
        return fake_model

    monkeypatch.setattr(cli_module, "OpenAIResponsesAgentModel", create_model)
    return fake_model


def sensitive_model_error() -> ModelExecutionError:
    try:
        raise RuntimeError("private prompt complete repository contents sk-secret")
    except RuntimeError as cause:
        try:
            raise ModelExecutionError("safe model request failure") from cause
        except ModelExecutionError as error:
            return error


def test_investigate_help_exposes_options_without_api_key() -> None:
    result = runner.invoke(cli_module.app, ["investigate", "--help"])

    assert result.exit_code == 0
    assert "--task" in result.stdout
    assert "--workspace" in result.stdout
    assert "--model" in result.stdout
    assert "--max-steps" in result.stdout
    assert "[default: 8]" in result.stdout
    assert "--api-key" not in result.stdout


def test_successful_investigation_prints_one_report_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task, workspace = cli_paths(tmp_path)
    model = install_fake_model(monkeypatch)
    calls: list[dict[str, object]] = []

    def run_fake(**kwargs: object) -> AgentState:
        calls.append(kwargs)
        return finished_state()

    monkeypatch.setattr(cli_module, "run_investigation_from_paths", run_fake)

    result = runner.invoke(cli_module.app, arguments(task, workspace))

    assert result.exit_code == 0
    assert result.stdout.count("RepoFix Read-Only Investigation") == 1
    assert "Phase: FINISHED" in result.stdout
    assert "Final summary: Read-only investigation complete." in result.stdout
    assert calls == [
        {
            "task_path": task,
            "workspace_root": workspace,
            "model": model,
            "max_steps": 8,
        }
    ]


def test_failed_terminal_state_prints_report_and_exits_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task, workspace = cli_paths(tmp_path)
    install_fake_model(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "run_investigation_from_paths",
        lambda **kwargs: failed_state(),
    )

    result = runner.invoke(cli_module.app, arguments(task, workspace))

    assert result.exit_code == 2
    assert "Phase: FAILED" in result.stdout
    assert "Failure reason: investigation exceeded the 8-step budget" in result.stdout


def test_investigate_rejects_missing_required_options() -> None:
    result = runner.invoke(cli_module.app, ["investigate"])

    assert result.exit_code == 2
    assert "Missing option" in result.stderr


def test_investigate_rejects_blank_model_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task, workspace = cli_paths(tmp_path)

    def reject_construction(*args: object, **kwargs: object) -> object:
        raise AssertionError("blank model reached model construction")

    monkeypatch.setattr(cli_module, "OpenAIResponsesAgentModel", reject_construction)
    args = arguments(task, workspace)
    args[args.index("configured-model")] = "   "

    result = runner.invoke(cli_module.app, args)

    assert result.exit_code == 1
    assert "model name must be nonempty" in result.stderr


@pytest.mark.parametrize("value", ["0", "21", "1.5"])
def test_investigate_rejects_invalid_step_limits(
    tmp_path: Path, value: str
) -> None:
    task, workspace = cli_paths(tmp_path)

    result = runner.invoke(
        cli_module.app,
        arguments(task, workspace, "--max-steps", value),
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.stderr


@pytest.mark.parametrize(
    "error",
    [
        TaskSpecLoadError("safe task loading failure"),
        sensitive_model_error(),
        ToolExecutionError("safe workspace failure"),
        AgentProtocolError("action is not permitted in the current phase"),
    ],
    ids=["task-loading", "model-execution", "repository-tool", "agent-protocol"],
)
def test_expected_operational_failures_are_sanitized_and_exit_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
) -> None:
    task, workspace = cli_paths(tmp_path)
    install_fake_model(monkeypatch)

    def fail(**kwargs: object) -> AgentState:
        raise error

    monkeypatch.setattr(cli_module, "run_investigation_from_paths", fail)

    result = runner.invoke(cli_module.app, arguments(task, workspace))

    assert result.exit_code == 1
    assert str(error) in result.stderr
    assert "private prompt" not in result.stderr
    assert "complete repository contents" not in result.stderr
    assert "sk-secret" not in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_programmer_errors_are_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task, workspace = cli_paths(tmp_path)
    install_fake_model(monkeypatch)

    def fail(**kwargs: object) -> AgentState:
        raise AssertionError("internal invariant failed")

    monkeypatch.setattr(cli_module, "run_investigation_from_paths", fail)

    result = runner.invoke(cli_module.app, arguments(task, workspace))

    assert result.exit_code == 1
    assert isinstance(result.exception, AssertionError)
    assert str(result.exception) == "internal invariant failed"


def test_unexpected_type_errors_are_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task, workspace = cli_paths(tmp_path)
    install_fake_model(monkeypatch)

    def fail(**kwargs: object) -> AgentState:
        raise TypeError("unexpected internal type error")

    monkeypatch.setattr(cli_module, "run_investigation_from_paths", fail)

    result = runner.invoke(cli_module.app, arguments(task, workspace))

    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)
    assert str(result.exception) == "unexpected internal type error"
