"""Integration coverage for deterministic fixture reproduction verification."""

from pathlib import Path
import shutil

import pytest

import repofix.models.openai_agent as openai_agent_module
from repofix.execution import CommandTerminationReason, LocalApprovedCommandExecutor
from repofix.reproduction import (
    ReproductionEvidence,
    ReproductionStatus,
    ReproductionTerminationReason,
    verify_reproduction,
)
from repofix.tasks import AgentTaskSpec, load_reproduction_task_bundle


def test_checked_in_fixture_is_reproduced_without_model_or_repository_mutation(
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
    bundle = load_reproduction_task_bundle(
        repository_root / "examples/reproduction/empty-header-bug.yaml"
    )
    task = bundle.agent_view()
    source = workspace / "src/header_parser.py"
    test_file = workspace / "tests/test_header_parser.py"
    before = {source: source.read_bytes(), test_file: test_file.read_bytes()}
    fake_key = "reproduction-test-api-key"
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    (workspace / "conftest.py").write_text(
        "import os\n\n"
        "def pytest_sessionstart(session):\n"
        "    assert 'OPENAI_API_KEY' not in os.environ\n",
        encoding="utf-8",
    )

    def reject_openai_construction(*args: object, **kwargs: object) -> object:
        raise AssertionError("reproduction verification attempted an OpenAI request")

    monkeypatch.setattr(openai_agent_module, "OpenAI", reject_openai_construction)
    command_executor = LocalApprovedCommandExecutor(
        workspace_root=workspace,
        approved_commands=task.approved_commands,
        timeout_seconds=task.timeout_seconds,
    )

    execution_result = command_executor.execute(bundle.reproduction.command_id)
    evidence = ReproductionEvidence.from_execution_result(execution_result)
    verdict = verify_reproduction(
        expectation=bundle.reproduction,
        evidence=evidence,
    )

    assert execution_result.termination_reason is CommandTerminationReason.COMPLETED
    assert execution_result.exit_code == 1
    assert verdict.status is ReproductionStatus.REPRODUCED
    assert verdict.matched_required_fragment_ids == tuple(
        sorted(fragment.fragment_id for fragment in bundle.reproduction.required_fragments)
    )
    assert verdict.missing_required_fragment_ids == ()
    assert verdict.forbidden_fragment_ids_found == ()
    output = f"{execution_result.stdout}\n{execution_result.stderr}"
    assert "1 failed, 1 passed" in output
    assert "ModuleNotFoundError" not in output
    assert "ERROR collecting" not in output
    assert fake_key not in output
    assert {path: path.read_bytes() for path in before} == before
    assert not (workspace / ".pytest_cache").exists()
    assert not any(workspace.rglob("__pycache__"))

    agent_serialized = task.model_dump()
    evaluator_rendered = repr(bundle.model_dump())
    verdict_rendered = repr(verdict.model_dump())
    assert type(task) is AgentTaskSpec
    assert "reproduction" not in repr(agent_serialized)
    assert "hidden_tests" not in evaluator_rendered
    assert "gold_patch" not in evaluator_rendered
    assert "patch" not in verdict_rendered


def test_unrelated_exit_one_evidence_is_inconclusive() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    bundle = load_reproduction_task_bundle(
        repository_root / "examples/reproduction/empty-header-bug.yaml"
    )
    unrelated_output = "FAILED tests/test_other.py - AssertionError: unrelated failure\n"
    evidence = ReproductionEvidence(
        command_id=bundle.reproduction.command_id,
        argv=("pytest", "-q", "-p", "no:cacheprovider"),
        termination_reason=ReproductionTerminationReason.COMPLETED,
        exit_code=1,
        stdout=unrelated_output,
        stderr="",
        stdout_bytes=len(unrelated_output.encode("utf-8")),
        stderr_bytes=0,
        had_decode_errors=False,
    )

    verdict = verify_reproduction(
        expectation=bundle.reproduction,
        evidence=evidence,
    )

    assert verdict.status is ReproductionStatus.INCONCLUSIVE
    assert verdict.matched_required_fragment_ids == ()
    assert verdict.missing_required_fragment_ids
