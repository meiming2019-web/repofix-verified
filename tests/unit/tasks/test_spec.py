"""Tests for validated task specification models."""

from typing import Any

import pytest
from pydantic import ValidationError

from repofix.tasks import (
    AgentTaskSpec,
    ApprovedCommand,
    EvaluatorTaskBundle,
    GoldPatchSpec,
    HiddenTestSpec,
)


FULL_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def valid_task_data() -> dict[str, Any]:
    """Return independent valid input data for an agent task."""
    return {
        "task_id": "task-001",
        "repository_url": "https://github.com/example/project.git",
        "pre_fix_commit": FULL_COMMIT,
        "issue_title": "Fix the failing behavior",
        "issue_body": "The command produces an incorrect result.",
        "approved_commands": {
            "unit_tests": {"argv": ["pytest", "-q"]},
        },
        "allowed_source_paths": ["src/repofix", "tests/unit"],
        "timeout_seconds": 300,
    }


def test_valid_approved_command() -> None:
    command = ApprovedCommand(argv=("python", "-m", "pytest"))

    assert command.argv == ("python", "-m", "pytest")


def test_approved_command_accepts_list_and_stores_tuple() -> None:
    command = ApprovedCommand.model_validate({"argv": ["pytest", "-q"]})

    assert command.argv == ("pytest", "-q")
    assert isinstance(command.argv, tuple)


def test_approved_command_rejects_empty_argv() -> None:
    with pytest.raises(ValidationError):
        ApprovedCommand(argv=())


def test_approved_command_rejects_shell_command_string() -> None:
    with pytest.raises(ValidationError):
        ApprovedCommand.model_validate({"argv": "pytest -q"})


@pytest.mark.parametrize("argv", [{"pytest": "-q"}, {"pytest", "-q"}])
def test_approved_command_rejects_mapping_and_set_containers(argv: object) -> None:
    with pytest.raises(ValidationError):
        ApprovedCommand.model_validate({"argv": argv})


def test_approved_command_rejects_generator_container() -> None:
    argv = (argument for argument in ["pytest", "-q"])

    with pytest.raises(ValidationError):
        ApprovedCommand.model_validate({"argv": argv})


@pytest.mark.parametrize("argument", [1, True, None])
def test_approved_command_rejects_non_string_elements(argument: object) -> None:
    with pytest.raises(ValidationError):
        ApprovedCommand.model_validate({"argv": ["pytest", argument]})


@pytest.mark.parametrize("executable", ["", "   "])
def test_approved_command_rejects_empty_executable(executable: str) -> None:
    with pytest.raises(ValidationError):
        ApprovedCommand.model_validate({"argv": [executable, "-q"]})


def test_approved_command_allows_empty_later_argument() -> None:
    command = ApprovedCommand.model_validate({"argv": ["python", ""]})

    assert command.argv == ("python", "")


def test_approved_command_rejects_nul_bytes() -> None:
    with pytest.raises(ValidationError):
        ApprovedCommand(argv=("python", "bad\0argument"))


def test_valid_agent_task_spec() -> None:
    task = AgentTaskSpec.model_validate(valid_task_data())

    assert task.task_id == "task-001"
    assert task.approved_commands["unit_tests"].argv == ("pytest", "-q")
    assert task.allowed_source_paths == ("src/repofix", "tests/unit")


def test_agent_task_accepts_yaml_style_nested_data() -> None:
    task = AgentTaskSpec.model_validate(valid_task_data())

    assert isinstance(task.approved_commands["unit_tests"], ApprovedCommand)
    assert task.approved_commands["unit_tests"].argv == ("pytest", "-q")
    assert isinstance(task.allowed_source_paths, tuple)


def test_agent_task_rejects_http_repository_url() -> None:
    data = valid_task_data()
    data["repository_url"] = "http://github.com/example/project.git"

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


@pytest.mark.parametrize(
    "repository_url",
    [
        "https://user@github.com/example/project.git",
        "https://user:password@github.com/example/project.git",
    ],
)
def test_agent_task_rejects_repository_credentials(repository_url: str) -> None:
    data = valid_task_data()
    data["repository_url"] = repository_url

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


@pytest.mark.parametrize(
    "commit",
    [
        "0123456",
        "g" * 40,
        "0" * 39,
        "0" * 41,
    ],
)
def test_agent_task_rejects_abbreviated_or_malformed_commit(commit: str) -> None:
    data = valid_task_data()
    data["pre_fix_commit"] = commit

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


def test_agent_task_rejects_empty_command_mapping() -> None:
    data = valid_task_data()
    data["approved_commands"] = {}

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


@pytest.mark.parametrize(
    "name",
    ["", "-", "_", "-tests", "_tests", "UnitTests", "unit tests", "test.command"],
)
def test_agent_task_rejects_invalid_command_names(name: str) -> None:
    data = valid_task_data()
    data["approved_commands"] = {name: {"argv": ["pytest"]}}

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


@pytest.mark.parametrize("name", ["unit_tests", "baseline-tests", "test1"])
def test_agent_task_accepts_valid_command_names(name: str) -> None:
    data = valid_task_data()
    data["approved_commands"] = {name: {"argv": ["pytest"]}}

    task = AgentTaskSpec.model_validate(data)

    assert name in task.approved_commands


def test_agent_task_rejects_absolute_paths() -> None:
    data = valid_task_data()
    data["allowed_source_paths"] = ("/src/repofix",)

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


@pytest.mark.parametrize("path", ["../src", "src/../tests"])
def test_agent_task_rejects_path_traversal(path: str) -> None:
    data = valid_task_data()
    data["allowed_source_paths"] = (path,)

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


def test_agent_task_rejects_backslash_separated_paths() -> None:
    data = valid_task_data()
    data["allowed_source_paths"] = (r"src\repofix",)

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


def test_agent_task_rejects_nul_bytes_in_paths() -> None:
    data = valid_task_data()
    data["allowed_source_paths"] = ["src/repofix\0hidden"]

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


@pytest.mark.parametrize("paths", ["src/repofix", {"src/repofix"}, {"src": "repofix"}])
def test_agent_task_rejects_invalid_path_containers(paths: object) -> None:
    data = valid_task_data()
    data["allowed_source_paths"] = paths

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


def test_agent_task_rejects_generator_path_container() -> None:
    data = valid_task_data()
    data["allowed_source_paths"] = (path for path in ["src/repofix"])

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


@pytest.mark.parametrize("path", [1, True, None])
def test_agent_task_rejects_non_string_path_elements(path: object) -> None:
    data = valid_task_data()
    data["allowed_source_paths"] = ["src/repofix", path]

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


@pytest.mark.parametrize("path", ["", ".", "src//repofix", "src/./repofix", "src/"])
def test_agent_task_rejects_empty_or_redundant_path_components(path: str) -> None:
    data = valid_task_data()
    data["allowed_source_paths"] = (path,)

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


@pytest.mark.parametrize("timeout", [True, 0, -1, 1.5, "30", 3601])
def test_agent_task_rejects_invalid_timeout_values(timeout: object) -> None:
    data = valid_task_data()
    data["timeout_seconds"] = timeout

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ApprovedCommand.model_validate({"argv": ("pytest",), "shell": True})


def test_valid_evaluator_task_bundle() -> None:
    task = AgentTaskSpec.model_validate(valid_task_data())
    bundle = EvaluatorTaskBundle(
        task=task,
        hidden_tests=HiddenTestSpec(
            commands={"hidden_tests": ApprovedCommand(argv=("pytest", "tests/hidden"))}
        ),
        gold_patch=GoldPatchSpec(patch="diff --git a/file.py b/file.py\n"),
    )

    assert bundle.task == task


def test_agent_view_excludes_evaluator_only_information() -> None:
    task = AgentTaskSpec.model_validate(valid_task_data())
    bundle = EvaluatorTaskBundle(
        task=task,
        hidden_tests=HiddenTestSpec(
            commands={"secret_check": ApprovedCommand(argv=("pytest", "secret_test.py"))}
        ),
        gold_patch=GoldPatchSpec(patch="SECRET PATCH CONTENT"),
    )

    agent_view = bundle.agent_view()
    serialized = agent_view.model_dump()

    assert agent_view is task
    assert set(serialized) == set(AgentTaskSpec.model_fields)
    assert "hidden_tests" not in serialized
    assert "gold_patch" not in serialized
    assert "secret_check" not in repr(serialized)
    assert "SECRET PATCH CONTENT" not in repr(serialized)


@pytest.mark.parametrize("field", ["hidden_tests", "gold_patch"])
def test_agent_task_rejects_evaluator_only_fields(field: str) -> None:
    data = valid_task_data()
    data[field] = {}

    with pytest.raises(ValidationError):
        AgentTaskSpec.model_validate(data)
