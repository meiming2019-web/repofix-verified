"""End-to-end post-patch reproduction verification for the checked-in fixture."""

from dataclasses import dataclass
from pathlib import Path
import shutil

import pytest

import repofix.runners.post_patch_reproduction as post_patch_runner_module
from repofix.agent import (
    AgentAction,
    AgentState,
    IssueUnderstanding,
    ReadFileAction,
    RecordHypothesisAction,
    RepairHypothesis,
    RunApprovedCommandAction,
    UnderstandIssueAction,
)
from repofix.execution import (
    ApprovedCommandExecutionResult,
    CommandTerminationReason,
    LocalApprovedCommandExecutor,
)
from repofix.patching import PatchProposalContext, PatchProposalDraft
from repofix.reproduction import (
    PostPatchReproductionStatus,
    ReproductionStatus,
    verify_post_patch_reproduction,
)
from repofix.runners import (
    run_patch_application_from_paths,
    run_patch_proposal_from_paths,
    run_post_patch_reproduction_from_paths,
    run_reproduction_from_paths,
)
from repofix.tasks import AgentTaskSpec, load_reproduction_task_bundle


class ReproductionScript:
    def __init__(self) -> None:
        self.call_count = 0

    def next_action(self, *, task: AgentTaskSpec, state: AgentState) -> AgentAction:
        actions: tuple[AgentAction, ...] = (
            UnderstandIssueAction(
                kind="understand_issue",
                understanding=IssueUnderstanding(
                    expected_behavior="Empty headers retain the configured value.",
                    observed_behavior="Empty headers return the default.",
                    reproduction_clues=("empty header",),
                    likely_components=("src/header_parser.py",),
                    missing_information=(),
                ),
            ),
            ReadFileAction(
                kind="read_file",
                path="src/header_parser.py",
                start_line=1,
                end_line=12,
            ),
            RecordHypothesisAction(
                kind="record_hypothesis",
                hypothesis=RepairHypothesis(
                    hypothesis_id="premature-default-return",
                    description="The empty-header branch returns the default too early.",
                    supporting_evidence=("The trusted source read identifies the branch.",),
                    contradicting_evidence=(),
                    confidence=0.95,
                    status="supported",
                ),
            ),
            RunApprovedCommandAction(command_id="unit_tests"),
        )
        action = actions[self.call_count]
        self.call_count += 1
        return action


class PatchScript:
    def __init__(self, *, line: int, replacement: str, rationale: str) -> None:
        self.line = line
        self.replacement = replacement
        self.rationale = rationale
        self.call_count = 0

    def propose_patch(self, *, context: PatchProposalContext) -> PatchProposalDraft:
        self.call_count += 1
        return PatchProposalDraft.model_validate(
            {
                "hypothesis_id": "premature-default-return",
                "model_summary": "Apply one bounded change without a verification claim.",
                "edits": [
                    {
                        "path": "src/header_parser.py",
                        "start_line": self.line,
                        "end_line": self.line,
                        "replacement_text": self.replacement,
                        "rationale": self.rationale,
                    }
                ],
            }
        )


@dataclass
class AppliedFixture:
    task_path: Path
    workspace: Path
    source: Path
    test_file: Path
    reproduction_model: ReproductionScript
    patch_model: PatchScript
    reproduction_result: object
    proposal: object
    application_result: object


def _applied_fixture(
    tmp_path: Path, *, line: int, replacement: str, rationale: str
) -> AppliedFixture:
    root = Path(__file__).resolve().parents[3]
    task_path = root / "examples/reproduction/empty-header-bug.yaml"
    workspace = tmp_path / "fixture"
    shutil.copytree(
        root / "examples/fixtures/empty-header-bug",
        workspace,
        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"),
    )
    reproduction_model = ReproductionScript()
    reproduction_result = run_reproduction_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        model=reproduction_model,
        max_steps=4,
    )
    patch_model = PatchScript(line=line, replacement=replacement, rationale=rationale)
    proposal = run_patch_proposal_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        reproduction_result=reproduction_result,
        model=patch_model,
    )
    application_result = run_patch_application_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        reproduction_result=reproduction_result,
        proposal=proposal,
    )
    return AppliedFixture(
        task_path=task_path,
        workspace=workspace,
        source=workspace / "src/header_parser.py",
        test_file=workspace / "tests/test_header_parser.py",
        reproduction_model=reproduction_model,
        patch_model=patch_model,
        reproduction_result=reproduction_result,
        proposal=proposal,
        application_result=application_result,
    )


def _run_real_post_patch(
    monkeypatch: pytest.MonkeyPatch, fixture: AppliedFixture
):
    calls: list[str] = []

    class CountingExecutor(LocalApprovedCommandExecutor):
        def execute(self, command_id: str) -> ApprovedCommandExecutionResult:
            calls.append(command_id)
            return super().execute(command_id)

    monkeypatch.setattr(
        post_patch_runner_module,
        "LocalApprovedCommandExecutor",
        CountingExecutor,
    )
    result = run_post_patch_reproduction_from_paths(
        task_path=fixture.task_path,
        workspace_root=fixture.workspace,
        original_reproduction_result=fixture.reproduction_result,  # type: ignore[arg-type]
        proposal=fixture.proposal,  # type: ignore[arg-type]
        application_result=fixture.application_result,  # type: ignore[arg-type]
    )
    return result, calls


def test_real_fixture_original_behavior_is_not_reproduced_after_patch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _applied_fixture(
        tmp_path,
        line=9,
        replacement="        return configured_value\n",
        rationale="Preserve the configured value for empty headers.",
    )
    source_after_application = fixture.source.read_bytes()
    test_after_application = fixture.test_file.read_bytes()
    model_calls = (fixture.reproduction_model.call_count, fixture.patch_model.call_count)

    result, command_calls = _run_real_post_patch(monkeypatch, fixture)

    assert command_calls == ["unit_tests"]
    assert result.evidence.exit_code == 0
    assert "2 passed" in result.evidence.stdout
    assert "test_empty_header_retains_configured_value" not in result.evidence.stdout
    assert result.verifier_verdict.status is ReproductionStatus.NOT_REPRODUCED
    assert (
        result.status
        is PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
    )
    assert fixture.source.read_bytes() == source_after_application
    assert fixture.test_file.read_bytes() == test_after_application
    assert (fixture.reproduction_model.call_count, fixture.patch_model.call_count) == model_calls
    assert result.command_id == "unit_tests"
    assert "verdict" not in result.model_dump(exclude={"verifier_verdict"})
    assert "regression and hidden verification have not run" in (
        result.verification_summary.lower()
    )
    assert not list(fixture.workspace.rglob(".repofix-patch-*"))


def test_real_fixture_original_behavior_is_still_reproduced_after_irrelevant_patch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _applied_fixture(
        tmp_path,
        line=1,
        replacement='"""Updated parser module documentation."""\n',
        rationale="Make a bounded source-only documentation change.",
    )
    source_after_application = fixture.source.read_bytes()
    model_calls = fixture.patch_model.call_count

    result, command_calls = _run_real_post_patch(monkeypatch, fixture)

    assert command_calls == ["unit_tests"]
    assert result.evidence.exit_code == 1
    assert "test_empty_header_retains_configured_value" in (
        result.evidence.stdout + result.evidence.stderr
    )
    assert result.verifier_verdict.status is ReproductionStatus.REPRODUCED
    assert (
        result.status
        is PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_STILL_REPRODUCED
    )
    assert fixture.source.read_bytes() == source_after_application
    assert fixture.patch_model.call_count == model_calls == 1


def test_applied_fixture_timeout_is_inconclusive_without_retry(tmp_path: Path) -> None:
    fixture = _applied_fixture(
        tmp_path,
        line=9,
        replacement="        return configured_value\n",
        rationale="Preserve the configured value for empty headers.",
    )
    bundle = load_reproduction_task_bundle(fixture.task_path)

    class TimeoutGateway:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def execute(self, command_id: str) -> ApprovedCommandExecutionResult:
            self.calls.append(command_id)
            return ApprovedCommandExecutionResult(
                command_id=command_id,
                argv=bundle.task.approved_commands[command_id].argv,
                termination_reason=CommandTerminationReason.TIMED_OUT,
                exit_code=None,
                stdout="partial output\n",
                stderr="",
                stdout_bytes=15,
                stderr_bytes=0,
                had_decode_errors=False,
            )

    gateway = TimeoutGateway()
    result = verify_post_patch_reproduction(
        workspace_root=fixture.workspace,
        task=bundle.agent_view(),
        expectation=bundle.reproduction,
        original_reproduction_result=fixture.reproduction_result,  # type: ignore[arg-type]
        proposal=fixture.proposal,  # type: ignore[arg-type]
        application_result=fixture.application_result,  # type: ignore[arg-type]
        command_gateway=gateway,
    )

    assert gateway.calls == ["unit_tests"]
    assert result.verifier_verdict.status is ReproductionStatus.INCONCLUSIVE
    assert result.status is PostPatchReproductionStatus.INCONCLUSIVE
