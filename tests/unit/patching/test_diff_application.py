"""Exact application coverage for generated unified diffs."""

import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from repofix.patching import PatchProposalDraft, validate_patch_proposal


def _bind_current_source_hash(reproduced_result, source: Path):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    observations = tuple(
        observation.model_copy(update={"full_file_sha256": digest})
        if observation.tool_name == "read_file"
        and observation.arguments.get("path") == "src/app.py"
        else observation
        for observation in reproduced_result.state.observations
    )
    state = reproduced_result.state.model_copy(update={"observations": observations})
    return reproduced_result.model_copy(update={"state": state})


@pytest.mark.parametrize(
    ("original", "edits", "candidate", "requires_no_newline_marker"),
    [
        (
            b"one\ntwo\ndelete\nkeep\n",
            [
                {
                    "path": "src/app.py",
                    "start_line": 1,
                    "end_line": 2,
                    "replacement_text": "ONE\nTWO\n",
                    "rationale": "multiline replacement",
                },
                {
                    "path": "src/app.py",
                    "start_line": 3,
                    "end_line": 3,
                    "replacement_text": "",
                    "rationale": "delete obsolete line",
                },
            ],
            b"ONE\nTWO\nkeep\n",
            False,
        ),
        (
            b"one\r\ntwo\r\n",
            [
                {
                    "path": "src/app.py",
                    "start_line": 2,
                    "end_line": 2,
                    "replacement_text": "TWO\n",
                    "rationale": "preserve CRLF",
                }
            ],
            b"one\r\nTWO\r\n",
            False,
        ),
        (
            b"one\ntwo",
            [
                {
                    "path": "src/app.py",
                    "start_line": 2,
                    "end_line": 2,
                    "replacement_text": "TWO",
                    "rationale": "preserve missing final newline",
                }
            ],
            b"one\nTWO",
            True,
        ),
    ],
    ids=["lf-multiline-and-deletion", "crlf", "no-final-newline"],
)
def test_generated_diff_passes_git_apply_and_produces_exact_candidate_bytes(
    patch_workspace,
    patch_task,
    reproduced_result,
    original: bytes,
    edits: list[dict[str, object]],
    candidate: bytes,
    requires_no_newline_marker: bool,
) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("system Git executable is unavailable")
    source = patch_workspace / "src/app.py"
    source.write_bytes(original)
    reproduced_result = _bind_current_source_hash(reproduced_result, source)
    draft = PatchProposalDraft.model_validate(
        {"hypothesis_id": "h1", "model_summary": "bounded edit", "edits": edits}
    )

    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=draft,
    )
    patch_bytes = proposal.unified_diff.encode("utf-8")

    subprocess.run(
        [git, "init", "-q"],
        cwd=patch_workspace,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    check = subprocess.run(
        [git, "apply", "--check", "-"],
        cwd=patch_workspace,
        input=patch_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert check.returncode == 0, check.stderr.decode("utf-8", errors="replace")
    subprocess.run(
        [git, "apply", "-"],
        cwd=patch_workspace,
        input=patch_bytes,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert source.read_bytes() == candidate
    assert ("\\ No newline at end of file" in proposal.unified_diff) is (requires_no_newline_marker)
