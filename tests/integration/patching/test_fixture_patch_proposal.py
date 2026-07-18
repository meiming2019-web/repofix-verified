from pathlib import Path
import hashlib
import shutil
import subprocess

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
from repofix.patching import (
    PatchProposalDraft,
    PatchProposalValidationError,
    validate_patch_proposal,
)
from repofix.reproduction import (
    ReproductionEvidence,
    ReproductionStatus,
    ReproductionTerminationReason,
    ReproductionVerdict,
    compute_reproduction_expectation_fingerprint,
)
from repofix.tasks import load_reproduction_task_bundle
from repofix.runners import run_patch_proposal_from_paths


class NeverPatchModel:
    def __init__(self) -> None:
        self.call_count = 0

    def propose_patch(self, *, context):
        self.call_count += 1
        raise AssertionError("patch model must not be called")


def reproduced_result(bundle, workspace: Path) -> ReproductionAgentRunResult:
    task = bundle.agent_view()
    output = "1 failed, 1 passed\n"
    evidence = ReproductionEvidence(
        command_id="unit_tests",
        argv=("pytest", "-q", "-p", "no:cacheprovider"),
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
        reasons=("matched",),
        matched_required_fragment_ids=("expected-test-count",),
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
        task_id="empty-header-bug",
        phase=AgentPhase.FINISHED,
        issue_understanding=IssueUnderstanding(
            expected_behavior="empty headers retain configured value",
            observed_behavior="default returned",
            reproduction_clues=("empty header",),
            likely_components=("src/header_parser.py",),
            missing_information=(),
        ),
        hypotheses=(
            RepairHypothesis(
                hypothesis_id="premature-default-return",
                description="empty branch returns default",
                supporting_evidence=("source read",),
                contradicting_evidence=(),
                confidence=0.95,
                status="supported",
            ),
        ),
        observations=(
            ToolObservation(
                step_index=1,
                tool_name="read_file",
                arguments={"path": "src/header_parser.py", "start_line": 1, "end_line": 10},
                success=True,
                output="9:         return DEFAULT_VALUE\n",
                error=None,
                full_file_sha256=hashlib.sha256(
                    (workspace / "src/header_parser.py").read_bytes()
                ).hexdigest(),
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
            compute_reproduction_expectation_fingerprint(bundle.reproduction)
        ),
    )


def test_fixture_proposal_is_validated_without_application(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    workspace = tmp_path / "fixture"
    shutil.copytree(
        root / "examples/fixtures/empty-header-bug",
        workspace,
        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"),
    )
    bundle = load_reproduction_task_bundle(root / "examples/reproduction/empty-header-bug.yaml")
    source = workspace / "src/header_parser.py"
    test_file = workspace / "tests/test_header_parser.py"
    before = (source.read_bytes(), test_file.read_bytes())
    draft = PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "premature-default-return",
            "model_summary": "Use the configured value in the empty-header branch.",
            "edits": [
                {
                    "path": "src/header_parser.py",
                    "start_line": 9,
                    "end_line": 9,
                    "replacement_text": "        return configured_value\n",
                    "rationale": "The empty branch should preserve the configured value.",
                }
            ],
        }
    )

    proposal = validate_patch_proposal(
        workspace_root=workspace,
        task=bundle.agent_view(),
        reproduction_result=reproduced_result(bundle, workspace),
        draft=draft,
    )

    assert len(proposal.edits) == 1
    assert proposal.edits[0].path == "src/header_parser.py"
    assert proposal.edits[0].original_text == "        return DEFAULT_VALUE\n"
    assert "+        return configured_value" in proposal.unified_diff
    assert len(proposal.proposal_digest) == 64
    git = shutil.which("git")
    if git is None:
        pytest.skip("system Git executable is unavailable")
    check = subprocess.run(
        [git, "apply", "--check", "-"],
        cwd=workspace,
        input=proposal.unified_diff.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert check.returncode == 0, check.stderr.decode("utf-8", errors="replace")
    assert (source.read_bytes(), test_file.read_bytes()) == before
    assert not (workspace / ".pytest_cache").exists()
    assert not any(workspace.rglob("__pycache__"))
    assert "fixed" not in proposal.validation_summary.lower()


@pytest.mark.parametrize(
    ("old", "new", "count"),
    [
        ("    - 1\n", "    - 2\n", 1),
        (
            "text: test_empty_header_retains_configured_value",
            "text: changed_required_fragment",
            1,
        ),
        ("text: ModuleNotFoundError", "text: ChangedImportError", 1),
        ("stream: combined", "stream: stderr", 1),
    ],
    ids=["exit-codes", "required-text", "forbidden-text", "fragment-stream"],
)
def test_expectation_changes_reject_old_result_before_model_call(
    tmp_path: Path, old: str, new: str, count: int
) -> None:
    root = Path(__file__).resolve().parents[3]
    original_task_path = root / "examples/reproduction/empty-header-bug.yaml"
    workspace = tmp_path / "fixture"
    shutil.copytree(root / "examples/fixtures/empty-header-bug", workspace)
    bundle = load_reproduction_task_bundle(original_task_path)
    result = reproduced_result(bundle, workspace)
    changed_task_path = tmp_path / "changed.yaml"
    contents = original_task_path.read_text(encoding="utf-8")
    assert contents.count(old) >= count
    changed_task_path.write_text(contents.replace(old, new, count), encoding="utf-8")
    model = NeverPatchModel()

    with pytest.raises(ValueError, match="expectation fingerprint"):
        run_patch_proposal_from_paths(
            task_path=changed_task_path,
            workspace_root=workspace,
            reproduction_result=result,
            model=model,
        )

    assert model.call_count == 0


def test_task_mismatch_and_stale_source_reject_before_model_call(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    task_path = root / "examples/reproduction/empty-header-bug.yaml"
    workspace = tmp_path / "fixture"
    shutil.copytree(root / "examples/fixtures/empty-header-bug", workspace)
    bundle = load_reproduction_task_bundle(task_path)
    result = reproduced_result(bundle, workspace)

    wrong_state = result.state.model_copy(update={"task_id": "other-task"})
    wrong_result = result.model_copy(update={"state": wrong_state})
    model = NeverPatchModel()
    with pytest.raises(ValueError, match="does not belong"):
        run_patch_proposal_from_paths(
            task_path=task_path,
            workspace_root=workspace,
            reproduction_result=wrong_result,
            model=model,
        )
    assert model.call_count == 0

    (workspace / "src/header_parser.py").write_text(
        "def parse_header(header, configured_value):\n    return 'completely different'\n",
        encoding="utf-8",
    )
    with pytest.raises(PatchProposalValidationError, match="changed after"):
        run_patch_proposal_from_paths(
            task_path=task_path,
            workspace_root=workspace,
            reproduction_result=result,
            model=model,
        )
    assert model.call_count == 0
