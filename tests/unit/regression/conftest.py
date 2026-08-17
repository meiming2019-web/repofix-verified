"""Reusable complete artifact chain for regression unit tests."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from repofix.agent import (
    AgentPhase,
    AgentReproductionObservation,
    AgentState,
    AgentWorkflow,
    EvaluatorReproductionAttempt,
    IssueUnderstanding,
    RepairHypothesis,
    ReproductionAgentRunResult,
    ToolObservation,
)
from repofix.agent.reproduction_loop import compute_task_fingerprint
from repofix.agent.state import REPRODUCED_TERMINAL_SUMMARY
from repofix.execution import (
    ApprovedCommandExecutionResult,
    CommandTerminationReason,
)
from repofix.patching import (
    PatchApplicationResult,
    PatchProposalDraft,
    ValidatedPatchProposal,
    apply_validated_patch_proposal,
    validate_patch_proposal,
)
from repofix.regression import (
    RegressionBaselineResult,
    RegressionSpecification,
    establish_regression_baseline,
)
from repofix.reproduction import (
    PostPatchReproductionResult,
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionStatus,
    ReproductionTerminationReason,
    ReproductionVerdict,
    compute_reproduction_expectation_fingerprint,
    verify_post_patch_reproduction,
)
from repofix.tasks import AgentTaskSpec


class Gateway:
    def __init__(
        self,
        result: ApprovedCommandExecutionResult | None = None,
        *,
        error: BaseException | None = None,
        mutation: tuple[Path, bytes] | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.mutation = mutation
        self.calls: list[str] = []

    def execute(self, command_id: str) -> ApprovedCommandExecutionResult:
        self.calls.append(command_id)
        if self.mutation is not None:
            path, contents = self.mutation
            path.write_bytes(contents)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def execution(
    command_id: str,
    argv: tuple[str, ...],
    *,
    exit_code: int | None = 0,
    termination: CommandTerminationReason = CommandTerminationReason.COMPLETED,
    decode_errors: bool = False,
) -> ApprovedCommandExecutionResult:
    output = "command output\n"
    return ApprovedCommandExecutionResult(
        command_id=command_id,
        argv=argv,
        termination_reason=termination,
        exit_code=exit_code,
        stdout=output,
        stderr="",
        stdout_bytes=len(output),
        stderr_bytes=0,
        had_decode_errors=decode_errors,
    )


def original_result(
    task: AgentTaskSpec,
    expectation: ReproductionExpectation,
    source: Path,
) -> ReproductionAgentRunResult:
    output = "TARGET FAILURE\n"
    evidence = ReproductionEvidence(
        command_id="unit_tests",
        argv=task.approved_commands["unit_tests"].argv,
        termination_reason=ReproductionTerminationReason.COMPLETED,
        exit_code=1,
        stdout=output,
        stderr="",
        stdout_bytes=len(output),
        stderr_bytes=0,
        had_decode_errors=False,
    )
    verdict = ReproductionVerdict(
        status=ReproductionStatus.REPRODUCED,
        command_id="unit_tests",
        exit_code=1,
        reasons=("expected failing behavior was reproduced",),
        matched_required_fragment_ids=("target",),
        missing_required_fragment_ids=(),
        forbidden_fragment_ids_found=(),
    )
    public = AgentReproductionObservation(
        command_id="unit_tests",
        termination_reason=CommandTerminationReason.COMPLETED,
        exit_code=1,
        stdout=output,
        stderr="",
        stdout_bytes=len(output),
        stderr_bytes=0,
        had_decode_errors=False,
        status=ReproductionStatus.REPRODUCED,
    )
    state = AgentState(
        task_id=task.task_id,
        phase=AgentPhase.FINISHED,
        issue_understanding=IssueUnderstanding(
            expected_behavior="right",
            observed_behavior="wrong",
            reproduction_clues=("target",),
            likely_components=("src/app.py",),
            missing_information=(),
        ),
        hypotheses=(
            RepairHypothesis(
                hypothesis_id="h1",
                description="wrong return",
                supporting_evidence=("source read",),
                contradicting_evidence=(),
                confidence=0.9,
                status="supported",
            ),
        ),
        observations=(
            ToolObservation(
                step_index=1,
                tool_name="read_file",
                arguments={"path": "src/app.py", "start_line": 1, "end_line": 2},
                success=True,
                output="1: def value():\n2:     return 'wrong'\n",
                error=None,
                full_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            ),
        ),
        step_count=4,
        terminal_summary=REPRODUCED_TERMINAL_SUMMARY,
        failure_reason=None,
        workflow=AgentWorkflow.REPRODUCTION,
        reproduction_command_id="unit_tests",
        reproduction_observations=(public,),
    )
    return ReproductionAgentRunResult(
        state=state,
        attempts=(EvaluatorReproductionAttempt(evidence=evidence, verdict=verdict),),
        task_fingerprint=compute_task_fingerprint(task),
        reproduction_expectation_fingerprint=(
            compute_reproduction_expectation_fingerprint(expectation)
        ),
    )


@dataclass
class PreparedBaseline:
    workspace: Path
    source: Path
    task: AgentTaskSpec
    expectation: ReproductionExpectation
    specification: RegressionSpecification
    original: ReproductionAgentRunResult
    proposal: ValidatedPatchProposal


@pytest.fixture
def prepared_baseline(tmp_path: Path) -> PreparedBaseline:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/app.py"
    source.write_bytes(b"def value():\n    return 'wrong'\n")
    task = AgentTaskSpec.model_validate(
        {
            "task_id": "regression-task",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0" * 40,
            "issue_title": "Wrong return",
            "issue_body": "The target behavior fails.",
            "approved_commands": {
                "unit_tests": {"argv": ["pytest", "-q", "target"]},
                "regression_tests": {"argv": ["pytest", "-q", "regression"]},
            },
            "allowed_source_paths": ["src"],
            "patchable_source_paths": ["src"],
            "timeout_seconds": 30,
        }
    )
    expectation = ReproductionExpectation.model_validate(
        {
            "command_id": "unit_tests",
            "expected_exit_codes": [1],
            "required_fragments": [
                {"fragment_id": "target", "stream": "combined", "text": "TARGET FAILURE"}
            ],
        }
    )
    specification = RegressionSpecification(command_id="regression_tests")
    original = original_result(task, expectation, source)
    proposal = validate_patch_proposal(
        workspace_root=tmp_path,
        task=task,
        reproduction_result=original,
        draft=PatchProposalDraft.model_validate(
            {
                "hypothesis_id": "h1",
                "model_summary": "bounded change",
                "edits": [
                    {
                        "path": "src/app.py",
                        "start_line": 2,
                        "end_line": 2,
                        "replacement_text": "    return 'right'\n",
                        "rationale": "Correct the return.",
                    }
                ],
            }
        ),
    )
    return PreparedBaseline(
        workspace=tmp_path,
        source=source,
        task=task,
        expectation=expectation,
        specification=specification,
        original=original,
        proposal=proposal,
    )


@dataclass
class PreparedVerification(PreparedBaseline):
    baseline: RegressionBaselineResult
    application: PatchApplicationResult
    post_patch: PostPatchReproductionResult


@pytest.fixture
def prepared_verification(prepared_baseline: PreparedBaseline) -> PreparedVerification:
    item = prepared_baseline
    regression_argv = item.task.approved_commands[item.specification.command_id].argv
    baseline = establish_regression_baseline(
        workspace_root=item.workspace,
        task=item.task,
        reproduction_expectation=item.expectation,
        regression_specification=item.specification,
        original_reproduction_result=item.original,
        proposal=item.proposal,
        command_gateway=Gateway(
            execution(item.specification.command_id, regression_argv)
        ),
    )
    application = apply_validated_patch_proposal(
        workspace_root=item.workspace,
        task=item.task,
        reproduction_result=item.original,
        proposal=item.proposal,
    )
    reproduction_argv = item.task.approved_commands[item.expectation.command_id].argv
    post_patch = verify_post_patch_reproduction(
        workspace_root=item.workspace,
        task=item.task,
        expectation=item.expectation,
        original_reproduction_result=item.original,
        proposal=item.proposal,
        application_result=application,
        command_gateway=Gateway(
            execution(item.expectation.command_id, reproduction_argv)
        ),
    )
    return PreparedVerification(
        **item.__dict__,
        baseline=baseline,
        application=application,
        post_patch=post_patch,
    )
