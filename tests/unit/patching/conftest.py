import hashlib
from pathlib import Path

import pytest

from repofix.agent import (
    AgentPhase,
    AgentReproductionObservation,
    AgentState,
    AgentWorkflow,
    IssueUnderstanding,
    RepairHypothesis,
    ToolObservation,
)
from repofix.agent.reproduction_loop import (
    EvaluatorReproductionAttempt,
    ReproductionAgentRunResult,
    compute_task_fingerprint,
)
from repofix.agent.state import REPRODUCED_TERMINAL_SUMMARY
from repofix.execution import CommandTerminationReason
from repofix.reproduction import (
    ReproductionEvidence,
    ReproductionStatus,
    ReproductionTerminationReason,
    ReproductionVerdict,
)
from repofix.tasks import AgentTaskSpec


@pytest.fixture
def patch_task() -> AgentTaskSpec:
    return AgentTaskSpec.model_validate(
        {
            "task_id": "patch-task",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0" * 40,
            "issue_title": "Wrong return",
            "issue_body": "The source returns the wrong value.",
            "approved_commands": {"tests": {"argv": ["pytest"]}},
            "allowed_source_paths": ["src", "tests"],
            "patchable_source_paths": ["src"],
            "timeout_seconds": 30,
        }
    )


@pytest.fixture
def reproduced_result(
    patch_task: AgentTaskSpec, patch_workspace: Path
) -> ReproductionAgentRunResult:
    output = "TARGET FAILURE\n"
    evidence = ReproductionEvidence(
        command_id="tests",
        argv=("pytest",),
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
        command_id="tests",
        exit_code=1,
        reasons=("matched",),
        matched_required_fragment_ids=("target",),
        missing_required_fragment_ids=(),
        forbidden_fragment_ids_found=(),
    )
    observation = AgentReproductionObservation(
        command_id="tests",
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
        task_id="patch-task",
        phase=AgentPhase.FINISHED,
        issue_understanding=IssueUnderstanding(
            expected_behavior="right",
            observed_behavior="wrong",
            reproduction_clues=("case",),
            likely_components=("src/app.py",),
            missing_information=(),
        ),
        hypotheses=(
            RepairHypothesis(
                hypothesis_id="h1",
                description="wrong branch",
                supporting_evidence=("read",),
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
                full_file_sha256=hashlib.sha256(
                    (patch_workspace / "src/app.py").read_bytes()
                ).hexdigest(),
            ),
        ),
        step_count=4,
        terminal_summary=REPRODUCED_TERMINAL_SUMMARY,
        failure_reason=None,
        workflow=AgentWorkflow.REPRODUCTION,
        reproduction_command_id="tests",
        reproduction_observations=(observation,),
    )
    return ReproductionAgentRunResult(
        state=state,
        attempts=(EvaluatorReproductionAttempt(evidence=evidence, verdict=verdict),),
        task_fingerprint=compute_task_fingerprint(patch_task),
        reproduction_expectation_fingerprint="e" * 64,
    )


@pytest.fixture
def patch_workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/app.py").write_text("def value():\n    return 'wrong'\n", encoding="utf-8")
    (tmp_path / "tests/test_app.py").write_text("test data\n", encoding="utf-8")
    return tmp_path
