"""Fixture-level controlled patch application coverage."""

import hashlib
import importlib.util
from pathlib import Path
import shutil

from repofix.patching import PatchProposalContext, PatchProposalDraft
from repofix.runners import (
    run_patch_application_from_paths,
    run_patch_proposal_from_paths,
)
from repofix.tasks import load_reproduction_task_bundle


def _reproduced_result(bundle, workspace):
    helper_path = Path(__file__).with_name("test_fixture_patch_proposal.py")
    spec = importlib.util.spec_from_file_location("fixture_patch_application_helper", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reproduced_result(bundle, workspace)


class DeterministicPatchModel:
    def __init__(self) -> None:
        self.call_count = 0

    def propose_patch(self, *, context: PatchProposalContext) -> PatchProposalDraft:
        self.call_count += 1
        return PatchProposalDraft.model_validate(
            {
                "hypothesis_id": "premature-default-return",
                "model_summary": "Use the configured value without claiming verification.",
                "edits": [
                    {
                        "path": "src/header_parser.py",
                        "start_line": 9,
                        "end_line": 9,
                        "replacement_text": "        return configured_value\n",
                        "rationale": "Preserve the configured value for an empty header.",
                    }
                ],
            }
        )


def test_fixture_validated_proposal_is_applied_without_verification(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    task_path = root / "examples/reproduction/empty-header-bug.yaml"
    workspace = tmp_path / "fixture"
    shutil.copytree(
        root / "examples/fixtures/empty-header-bug",
        workspace,
        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"),
    )
    bundle = load_reproduction_task_bundle(task_path)
    reproduction_result = _reproduced_result(bundle, workspace)
    source = workspace / "src/header_parser.py"
    test_file = workspace / "tests/test_header_parser.py"
    original_source = source.read_bytes()
    original_test = test_file.read_bytes()
    before_files = {
        path.relative_to(workspace): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    model = DeterministicPatchModel()
    proposal = run_patch_proposal_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        reproduction_result=reproduction_result,
        model=model,
    )
    assert source.read_bytes() == original_source
    calls_before_application = model.call_count

    result = run_patch_application_from_paths(
        task_path=task_path,
        workspace_root=workspace,
        reproduction_result=reproduction_result,
        proposal=proposal,
    )

    expected_source = original_source.replace(
        b"        return DEFAULT_VALUE\n",
        b"        return configured_value\n",
    )
    assert source.read_bytes() == expected_source
    assert test_file.read_bytes() == original_test
    after_files = {
        path.relative_to(workspace): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    assert set(after_files) == set(before_files)
    assert {
        path for path in after_files if after_files[path] != before_files[path]
    } == {Path("src/header_parser.py")}
    snapshot = proposal.file_snapshots[0]
    applied = result.files[0]
    assert snapshot.original_file_sha256 == hashlib.sha256(original_source).hexdigest()
    assert snapshot.original_file_sha256 == applied.original_file_sha256
    assert applied.candidate_file_sha256 == snapshot.candidate_file_sha256
    assert applied.candidate_file_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert model.call_count == calls_before_application == 1
    assert result.status.value == "applied"
    assert result.application_summary.endswith("No verification tests have run.")
    assert "fixed" not in result.application_summary.lower()
    assert "verdict" not in result.model_dump()
    assert not list(workspace.rglob(".repofix-patch-*"))
    assert not (workspace / ".pytest_cache").exists()
    assert not any(workspace.rglob("__pycache__"))
