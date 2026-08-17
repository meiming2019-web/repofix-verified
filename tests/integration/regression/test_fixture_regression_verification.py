"""End-to-end regression baseline and verification for the checked-in fixture."""

from dataclasses import dataclass
from pathlib import Path
import shutil

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
)
from repofix.patching import PatchProposalContext, PatchProposalDraft
from repofix.regression import (
    RegressionBaselineStatus,
    RegressionVerificationStatus,
    verify_post_patch_regression,
)
from repofix.reproduction import PostPatchReproductionStatus
from repofix.runners import (
    run_patch_application_from_paths,
    run_patch_proposal_from_paths,
    run_post_patch_reproduction_from_paths,
    run_regression_baseline_from_paths,
    run_regression_verification_from_paths,
    run_reproduction_from_paths,
)
from repofix.tasks import AgentTaskSpec, load_reproduction_task_bundle


class ReproductionScript:
    def __init__(self) -> None:
        self.calls = 0

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
        action = actions[self.calls]
        self.calls += 1
        return action


class PatchScript:
    def __init__(self, *, start: int, end: int, replacement: str) -> None:
        self.start = start
        self.end = end
        self.replacement = replacement
        self.calls = 0

    def propose_patch(self, *, context: PatchProposalContext) -> PatchProposalDraft:
        self.calls += 1
        return PatchProposalDraft.model_validate(
            {
                "hypothesis_id": "premature-default-return",
                "model_summary": "Apply one bounded source correction.",
                "edits": [
                    {
                        "path": "src/header_parser.py",
                        "start_line": self.start,
                        "end_line": self.end,
                        "replacement_text": self.replacement,
                        "rationale": "Preserve the configured empty-header value.",
                    }
                ],
            }
        )


@dataclass
class Chain:
    task_path: Path
    workspace: Path
    reproduction_model: ReproductionScript
    patch_model: PatchScript
    original: object
    proposal: object
    baseline: object
    application: object
    post_patch: object


def _chain(tmp_path: Path, *, start: int, end: int, replacement: str) -> Chain:
    root = Path(__file__).resolve().parents[3]
    task_path = root / "examples/reproduction/empty-header-bug.yaml"
    workspace = tmp_path / "fixture"
    shutil.copytree(
        root / "examples/fixtures/empty-header-bug",
        workspace,
        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"),
    )
    reproduction_model = ReproductionScript()
    original = run_reproduction_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        model=reproduction_model,
        max_steps=4,
    )
    patch_model = PatchScript(start=start, end=end, replacement=replacement)
    proposal = run_patch_proposal_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        reproduction_result=original,
        model=patch_model,
    )
    baseline = run_regression_baseline_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        original_reproduction_result=original,
        proposal=proposal,
    )
    application = run_patch_application_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        reproduction_result=original,
        proposal=proposal,
    )
    post_patch = run_post_patch_reproduction_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        original_reproduction_result=original,
        proposal=proposal,
        application_result=application,
    )
    return Chain(
        task_path=task_path,
        workspace=workspace,
        reproduction_model=reproduction_model,
        patch_model=patch_model,
        original=original,
        proposal=proposal,
        baseline=baseline,
        application=application,
        post_patch=post_patch,
    )


def _verify(chain: Chain):
    return run_regression_verification_from_paths(
        task_path=chain.task_path,
        workspace_root=chain.workspace,
        original_reproduction_result=chain.original,  # type: ignore[arg-type]
        proposal=chain.proposal,  # type: ignore[arg-type]
        baseline_result=chain.baseline,  # type: ignore[arg-type]
        application_result=chain.application,  # type: ignore[arg-type]
        post_patch_reproduction_result=chain.post_patch,  # type: ignore[arg-type]
    )


def test_corrective_patch_passes_the_same_regression_command_before_and_after(
    tmp_path: Path,
) -> None:
    chain = _chain(
        tmp_path,
        start=9,
        end=9,
        replacement="        return configured_value\n",
    )
    source_before_regression = (chain.workspace / "src/header_parser.py").read_bytes()
    model_calls = (chain.reproduction_model.calls, chain.patch_model.calls)

    result = _verify(chain)

    assert chain.baseline.status is RegressionBaselineStatus.PASSED  # type: ignore[union-attr]
    assert (
        chain.post_patch.status  # type: ignore[union-attr]
        is PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
    )
    assert result.status is RegressionVerificationStatus.REGRESSION_COMMAND_PASSED
    assert result.evidence.exit_code == 0
    assert "1 passed" in result.evidence.stdout
    assert (chain.workspace / "src/header_parser.py").read_bytes() == source_before_regression
    assert (chain.reproduction_model.calls, chain.patch_model.calls) == model_calls == (4, 1)


def test_patch_fixing_target_but_breaking_unrelated_behavior_is_observed_as_failure(
    tmp_path: Path,
) -> None:
    chain = _chain(
        tmp_path,
        start=8,
        end=10,
        replacement=(
            '    if header == "":\n'
            "        return configured_value\n"
            "    return configured_value\n"
        ),
    )

    result = _verify(chain)

    assert chain.baseline.status is RegressionBaselineStatus.PASSED  # type: ignore[union-attr]
    assert (
        chain.post_patch.status  # type: ignore[union-attr]
        is PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
    )
    assert result.status is RegressionVerificationStatus.REGRESSION_COMMAND_FAILED
    assert result.evidence.exit_code == 1
    assert "test_nonempty_header_is_returned" in (
        result.evidence.stdout + result.evidence.stderr
    )
    assert chain.patch_model.calls == 1


def test_timeout_after_real_passing_baseline_is_inconclusive_without_retry(
    tmp_path: Path,
) -> None:
    chain = _chain(
        tmp_path,
        start=9,
        end=9,
        replacement="        return configured_value\n",
    )
    bundle = load_reproduction_task_bundle(chain.task_path)

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
    result = verify_post_patch_regression(
        workspace_root=chain.workspace,
        task=bundle.agent_view(),
        reproduction_expectation=bundle.reproduction,
        regression_specification=bundle.regression,
        original_reproduction_result=chain.original,  # type: ignore[arg-type]
        proposal=chain.proposal,  # type: ignore[arg-type]
        baseline_result=chain.baseline,  # type: ignore[arg-type]
        application_result=chain.application,  # type: ignore[arg-type]
        post_patch_reproduction_result=chain.post_patch,  # type: ignore[arg-type]
        command_gateway=gateway,
    )

    assert result.status is RegressionVerificationStatus.INCONCLUSIVE
    assert gateway.calls == ["regression_tests"]
