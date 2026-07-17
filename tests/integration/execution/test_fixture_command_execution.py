"""Integration test for the checked-in fixture's approved unit-test command."""

from pathlib import Path
import shutil

import pytest

import repofix.models.openai_agent as openai_agent_module
from repofix.execution import CommandTerminationReason, LocalApprovedCommandExecutor
from repofix.tasks import load_agent_task_spec


def test_fixture_approved_command_captures_raw_failing_test_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    source_workspace = repository_root / "examples/fixtures/empty-header-bug"
    workspace = tmp_path / "empty-header-bug"
    shutil.copytree(
        source_workspace,
        workspace,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    task = load_agent_task_spec(repository_root / "examples/tasks/empty-header-bug.yaml")
    source = workspace / "src/header_parser.py"
    test_file = workspace / "tests/test_header_parser.py"
    before = {source: source.read_bytes(), test_file: test_file.read_bytes()}
    fake_key = "not-a-real-openai-key"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    (workspace / "conftest.py").write_text(
        "import os\n\n"
        "def pytest_sessionstart(session):\n"
        "    assert 'OPENAI_API_KEY' not in os.environ\n",
        encoding="utf-8",
    )

    def reject_openai_construction(*args: object, **kwargs: object) -> object:
        raise AssertionError("fixture command execution attempted an OpenAI request")

    monkeypatch.setattr(openai_agent_module, "OpenAI", reject_openai_construction)
    command_executor = LocalApprovedCommandExecutor(
        workspace_root=workspace,
        approved_commands=task.approved_commands,
        timeout_seconds=task.timeout_seconds,
    )

    result = command_executor.execute("unit_tests")

    assert result.termination_reason is CommandTerminationReason.COMPLETED
    assert result.exit_code == 1
    assert result.argv == ("pytest", "-q", "-p", "no:cacheprovider")
    output = f"{result.stdout}\n{result.stderr}"
    assert "test_empty_header_retains_configured_value" in output
    assert "1 failed, 1 passed" in output
    assert "ERROR collecting" not in output
    assert "ModuleNotFoundError" not in output
    assert "patch was applied" not in output
    assert fake_key not in output
    assert {path: path.read_bytes() for path in before} == before
    assert not (workspace / ".pytest_cache").exists()
    assert not any(workspace.rglob("__pycache__"))

    rendered = repr(result.model_dump())
    assert "reproduced" not in rendered
    assert "hidden_tests" not in rendered
    assert "gold_patch" not in rendered
    assert "evaluator" not in rendered
