"""End-to-end runner coverage for the controlled proposal milestone."""

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
from repofix.patching import (
    PATCH_VALIDATION_SUMMARY,
    PatchProposalContext,
    PatchProposalDraft,
)
from repofix.runners import run_patch_proposal_from_paths, run_reproduction_from_paths
from repofix.tasks import AgentTaskSpec


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
    def __init__(self) -> None:
        self.call_count = 0
        self.contexts: list[PatchProposalContext] = []

    def propose_patch(self, *, context: PatchProposalContext) -> PatchProposalDraft:
        self.call_count += 1
        self.contexts.append(context)
        return PatchProposalDraft.model_validate(
            {
                "hypothesis_id": "premature-default-return",
                "model_summary": "Propose use of the configured value without a repair claim.",
                "edits": [
                    {
                        "path": "src/header_parser.py",
                        "start_line": 9,
                        "end_line": 9,
                        "replacement_text": "        return configured_value\n",
                        "rationale": "Preserve the configured value in the empty branch.",
                    }
                ],
            }
        )


def test_real_reproduction_to_patch_runner_happy_path(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    task_path = repository_root / "examples/reproduction/empty-header-bug.yaml"
    workspace = tmp_path / "fixture"
    shutil.copytree(
        repository_root / "examples/fixtures/empty-header-bug",
        workspace,
        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"),
    )
    source = workspace / "src/header_parser.py"
    tests = workspace / "tests/test_header_parser.py"
    before_bytes = (source.read_bytes(), tests.read_bytes())
    before_paths = {path.relative_to(workspace) for path in workspace.rglob("*") if path.is_file()}
    reproduction_model = ReproductionScript()

    reproduction_result = run_reproduction_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        model=reproduction_model,
        max_steps=4,
    )
    patch_model = PatchScript()
    proposal = run_patch_proposal_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        reproduction_result=reproduction_result,
        model=patch_model,
    )

    assert reproduction_model.call_count == 4
    assert patch_model.call_count == 1
    assert len(patch_model.contexts) == 1
    rendered_context = repr(patch_model.contexts[0].model_dump())
    for private in (
        "expected_exit_codes",
        "required_fragments",
        "forbidden_fragments",
        "reproduction_expectation_fingerprint",
        "full_file_sha256",
    ):
        assert private not in rendered_context
    assert proposal.validation_summary == PATCH_VALIDATION_SUMMARY
    assert proposal.validation_status.value == "structurally_validated_unapplied"
    assert "fixed" not in proposal.validation_summary.lower()
    assert "repaired" not in proposal.validation_summary.lower()
    assert "fixed" not in proposal.model_summary.lower()
    assert "verified" not in proposal.model_summary.lower()
    assert (source.read_bytes(), tests.read_bytes()) == before_bytes
    after_paths = {path.relative_to(workspace) for path in workspace.rglob("*") if path.is_file()}
    assert after_paths == before_paths
    assert not (workspace / ".pytest_cache").exists()
    assert not any(workspace.rglob("__pycache__"))
