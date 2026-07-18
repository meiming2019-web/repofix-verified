"""Tests for controlled application of validated patch proposals."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

import repofix.patching.application as application_module
from repofix.agent import ToolObservation
from repofix.patching import (
    PATCH_APPLICATION_SUMMARY,
    PatchApplicationError,
    PatchApplicationResult,
    PatchApplicationStatus,
    PatchProposalDraft,
    ValidatedPatchProposal,
    apply_validated_patch_proposal,
    compute_proposal_digest,
    validate_patch_proposal,
)


def _draft(
    *edits: dict[str, object], summary: str = "Apply a bounded correction."
) -> PatchProposalDraft:
    return PatchProposalDraft.model_validate(
        {"hypothesis_id": "h1", "model_summary": summary, "edits": edits}
    )


def _single_draft(replacement: str = "    return 'right'\n") -> PatchProposalDraft:
    return _draft(
        {
            "path": "src/app.py",
            "start_line": 2,
            "end_line": 2,
            "replacement_text": replacement,
            "rationale": "Use the intended return value.",
        }
    )


def _proposal(patch_workspace, patch_task, reproduced_result, draft=None):
    return validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=draft or _single_draft(),
    )


def _with_read(reproduced_result, workspace: Path, path: str):
    current_hash = hashlib.sha256((workspace / path).read_bytes()).hexdigest()
    existing = tuple(
        observation.model_copy(update={"full_file_sha256": current_hash})
        if observation.tool_name == "read_file" and observation.arguments.get("path") == path
        else observation
        for observation in reproduced_result.state.observations
    )
    if existing != reproduced_result.state.observations:
        state = reproduced_result.state.model_copy(update={"observations": existing})
        return reproduced_result.model_copy(update={"state": state})
    observation = ToolObservation(
        step_index=len(reproduced_result.state.observations) + 1,
        tool_name="read_file",
        arguments={"path": path, "start_line": 1, "end_line": 10},
        success=True,
        output="read",
        error=None,
        full_file_sha256=current_hash,
    )
    state = reproduced_result.state.model_copy(
        update={"observations": (*reproduced_result.state.observations, observation)}
    )
    return reproduced_result.model_copy(update={"state": state})


def _multi_file_setup(patch_workspace, reproduced_result):
    other = patch_workspace / "src/other.py"
    other.write_bytes(b"value = 'old'\n")
    result = _with_read(reproduced_result, patch_workspace, "src/other.py")
    draft = _draft(
        {
            "path": "src/app.py",
            "start_line": 2,
            "end_line": 2,
            "replacement_text": "    return 'right'\n",
            "rationale": "Correct the return.",
        },
        {
            "path": "src/other.py",
            "start_line": 1,
            "end_line": 1,
            "replacement_text": "value = 'new'\n",
            "rationale": "Update the related value.",
        },
    )
    return result, draft


def _recanonicalize(proposal: ValidatedPatchProposal, **updates: object):
    changed = proposal.model_copy(update=updates)
    digest = compute_proposal_digest(
        task_id=changed.task_id,
        task_fingerprint=changed.task_fingerprint,
        reproduction_expectation_fingerprint=changed.reproduction_expectation_fingerprint,
        reproduction_run_fingerprint=changed.reproduction_run_fingerprint,
        hypothesis_id=changed.hypothesis_id,
        model_summary=changed.model_summary,
        validation_status=changed.validation_status,
        validation_summary=changed.validation_summary,
        edits=changed.edits,
        file_snapshots=changed.file_snapshots,
        unified_diff=changed.unified_diff,
    )
    return ValidatedPatchProposal.model_validate(
        {**changed.model_dump(), "proposal_digest": digest}
    )


def _apply(patch_workspace, patch_task, reproduced_result, proposal):
    return apply_validated_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        proposal=proposal,
    )


def _temporary_paths(workspace: Path) -> list[Path]:
    return list(workspace.rglob(".repofix-patch-*"))


def test_valid_single_file_application_and_system_owned_result(
    patch_workspace, patch_task, reproduced_result
) -> None:
    proposal = _proposal(patch_workspace, patch_task, reproduced_result)

    result = _apply(patch_workspace, patch_task, reproduced_result, proposal)

    assert (patch_workspace / "src/app.py").read_bytes() == (
        b"def value():\n    return 'right'\n"
    )
    assert result.status is PatchApplicationStatus.APPLIED
    assert result.application_summary == PATCH_APPLICATION_SUMMARY
    assert "fixed" not in result.application_summary.lower()
    assert "pass" not in result.application_summary.lower()
    assert result.files[0].candidate_file_sha256 == hashlib.sha256(
        (patch_workspace / "src/app.py").read_bytes()
    ).hexdigest()
    assert result.proposal_digest == proposal.proposal_digest
    assert _temporary_paths(patch_workspace) == []
    with pytest.raises(ValidationError):
        PatchApplicationResult.model_validate({**result.model_dump(), "unexpected": True})


def test_multiple_edits_to_one_file_are_applied_bottom_to_top(
    patch_workspace, patch_task, reproduced_result
) -> None:
    source = patch_workspace / "src/app.py"
    source.write_bytes(b"first\nsecond\nthird\n")
    reproduced_result = _with_read(reproduced_result, patch_workspace, "src/app.py")
    draft = _draft(
        {
            "path": "src/app.py",
            "start_line": 1,
            "end_line": 1,
            "replacement_text": "FIRST\n",
            "rationale": "Uppercase first.",
        },
        {
            "path": "src/app.py",
            "start_line": 3,
            "end_line": 3,
            "replacement_text": "THIRD\n",
            "rationale": "Uppercase third.",
        },
    )
    proposal = _proposal(patch_workspace, patch_task, reproduced_result, draft)

    _apply(patch_workspace, patch_task, reproduced_result, proposal)

    assert source.read_bytes() == b"FIRST\nsecond\nTHIRD\n"


def test_bounded_multi_file_application(
    patch_workspace, patch_task, reproduced_result
) -> None:
    reproduced_result, draft = _multi_file_setup(patch_workspace, reproduced_result)
    proposal = _proposal(patch_workspace, patch_task, reproduced_result, draft)

    result = _apply(patch_workspace, patch_task, reproduced_result, proposal)

    assert tuple(item.path for item in result.files) == ("src/app.py", "src/other.py")
    assert (patch_workspace / "src/app.py").read_bytes().endswith(b"return 'right'\n")
    assert (patch_workspace / "src/other.py").read_bytes() == b"value = 'new'\n"


@pytest.mark.parametrize(
    ("original", "replacement", "candidate"),
    [
        (b"one\ntwo\n", "TWO\n", b"one\nTWO\n"),
        (b"one\r\ntwo\r\n", "TWO\n", b"one\r\nTWO\r\n"),
        (b"one\ntwo", "TWO", b"one\nTWO"),
    ],
    ids=["lf", "crlf", "no-final-newline"],
)
def test_supported_line_endings_apply_byte_for_byte(
    patch_workspace,
    patch_task,
    reproduced_result,
    original: bytes,
    replacement: str,
    candidate: bytes,
) -> None:
    source = patch_workspace / "src/app.py"
    source.write_bytes(original)
    reproduced_result = _with_read(reproduced_result, patch_workspace, "src/app.py")
    proposal = _proposal(
        patch_workspace,
        patch_task,
        reproduced_result,
        _draft(
            {
                "path": "src/app.py",
                "start_line": 2,
                "end_line": 2,
                "replacement_text": replacement,
                "rationale": "Change the second line.",
            }
        ),
    )

    _apply(patch_workspace, patch_task, reproduced_result, proposal)

    assert source.read_bytes() == candidate


@pytest.mark.parametrize("binding", ["task", "expectation", "run"])
def test_binding_mismatches_are_rejected_before_writes(
    patch_workspace, patch_task, reproduced_result, binding: str
) -> None:
    source = patch_workspace / "src/app.py"
    proposal = _proposal(patch_workspace, patch_task, reproduced_result)
    before = source.read_bytes()
    task = patch_task
    result = reproduced_result
    if binding == "task":
        task = patch_task.model_copy(update={"issue_body": "changed task"})
    elif binding == "expectation":
        result = reproduced_result.model_copy(
            update={"reproduction_expectation_fingerprint": "f" * 64}
        )
    else:
        understanding = reproduced_result.state.issue_understanding.model_copy(
            update={"observed_behavior": "changed run"}
        )
        state = reproduced_result.state.model_copy(update={"issue_understanding": understanding})
        result = reproduced_result.model_copy(update={"state": state})

    with pytest.raises(PatchApplicationError, match=f"{binding} fingerprint"):
        _apply(patch_workspace, task, result, proposal)

    assert source.read_bytes() == before


def test_proposal_digest_mismatch_is_rejected_before_writes(
    patch_workspace, patch_task, reproduced_result
) -> None:
    source = patch_workspace / "src/app.py"
    proposal = _proposal(patch_workspace, patch_task, reproduced_result)
    forged = proposal.model_copy(update={"proposal_digest": "0" * 64})
    before = source.read_bytes()

    with pytest.raises(PatchApplicationError, match="canonical integrity"):
        _apply(patch_workspace, patch_task, reproduced_result, forged)

    assert source.read_bytes() == before


@pytest.mark.parametrize("mismatch", ["hash", "size"])
def test_candidate_metadata_mismatch_is_rejected_before_writes(
    patch_workspace, patch_task, reproduced_result, mismatch: str
) -> None:
    source = patch_workspace / "src/app.py"
    proposal = _proposal(patch_workspace, patch_task, reproduced_result)
    snapshot = proposal.file_snapshots[0]
    update = (
        {"candidate_file_sha256": "f" * 64}
        if mismatch == "hash"
        else {"candidate_size_bytes": snapshot.candidate_size_bytes + 1}
    )
    forged = _recanonicalize(
        proposal,
        file_snapshots=(snapshot.model_copy(update=update),),
    )
    before = source.read_bytes()

    with pytest.raises(PatchApplicationError, match=f"candidate {mismatch}"):
        _apply(patch_workspace, patch_task, reproduced_result, forged)

    assert source.read_bytes() == before


def test_stale_current_source_rejects_all_files_before_writes(
    patch_workspace, patch_task, reproduced_result
) -> None:
    reproduced_result, draft = _multi_file_setup(patch_workspace, reproduced_result)
    proposal = _proposal(patch_workspace, patch_task, reproduced_result, draft)
    first = patch_workspace / "src/app.py"
    second = patch_workspace / "src/other.py"
    before_first = first.read_bytes()
    second.write_bytes(b"stale\n")
    before_second = second.read_bytes()

    with pytest.raises(PatchApplicationError, match="target hash"):
        _apply(patch_workspace, patch_task, reproduced_result, proposal)

    assert first.read_bytes() == before_first
    assert second.read_bytes() == before_second
    assert _temporary_paths(patch_workspace) == []


@pytest.mark.parametrize("target_kind", ["missing", "directory", "symlink"])
def test_invalid_current_targets_are_rejected_without_writes(
    patch_workspace, patch_task, reproduced_result, target_kind: str
) -> None:
    source = patch_workspace / "src/app.py"
    proposal = _proposal(patch_workspace, patch_task, reproduced_result)
    original = source.read_bytes()
    source.unlink()
    if target_kind == "directory":
        source.mkdir()
    elif target_kind == "symlink":
        external = patch_workspace.parent / "external.py"
        external.write_bytes(original)
        try:
            source.symlink_to(external)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(PatchApplicationError, match="safely inspected") as caught:
        _apply(patch_workspace, patch_task, reproduced_result, proposal)

    assert str(patch_workspace) not in str(caught.value)
    assert _temporary_paths(patch_workspace) == []


def test_target_outside_patchable_paths_is_rejected_before_inspection(
    patch_workspace, patch_task, reproduced_result
) -> None:
    proposal = _proposal(patch_workspace, patch_task, reproduced_result)
    edit = proposal.edits[0].model_copy(update={"path": "tests/test_app.py"})
    snapshot = proposal.file_snapshots[0].model_copy(update={"path": "tests/test_app.py"})
    forged = _recanonicalize(proposal, edits=(edit,), file_snapshots=(snapshot,))
    before = (patch_workspace / "tests/test_app.py").read_bytes()

    with pytest.raises(PatchApplicationError, match="outside patchable"):
        _apply(patch_workspace, patch_task, reproduced_result, forged)

    assert (patch_workspace / "tests/test_app.py").read_bytes() == before


def test_original_permission_mode_is_preserved(
    patch_workspace, patch_task, reproduced_result
) -> None:
    source = patch_workspace / "src/app.py"
    source.chmod(0o754)
    reproduced_result = _with_read(reproduced_result, patch_workspace, "src/app.py")
    proposal = _proposal(patch_workspace, patch_task, reproduced_result)

    _apply(patch_workspace, patch_task, reproduced_result, proposal)

    assert source.stat().st_mode & 0o777 == 0o754


def test_failure_before_first_replace_changes_nothing_and_cleans_temporaries(
    monkeypatch: pytest.MonkeyPatch, patch_workspace, patch_task, reproduced_result
) -> None:
    proposal = _proposal(patch_workspace, patch_task, reproduced_result)
    source = patch_workspace / "src/app.py"
    before = source.read_bytes()

    def fail_replace(source_path: object, target_path: object) -> None:
        raise PermissionError("replace denied")

    monkeypatch.setattr(application_module.os, "replace", fail_replace)
    with pytest.raises(PatchApplicationError, match="atomically replace"):
        _apply(patch_workspace, patch_task, reproduced_result, proposal)

    assert source.read_bytes() == before
    assert _temporary_paths(patch_workspace) == []


def test_second_replace_failure_rolls_back_first_and_cleans_temporaries(
    monkeypatch: pytest.MonkeyPatch, patch_workspace, patch_task, reproduced_result
) -> None:
    reproduced_result, draft = _multi_file_setup(patch_workspace, reproduced_result)
    proposal = _proposal(patch_workspace, patch_task, reproduced_result, draft)
    paths = [patch_workspace / "src/app.py", patch_workspace / "src/other.py"]
    before = [path.read_bytes() for path in paths]
    original_replace = application_module.os.replace
    calls = 0

    def fail_second(source_path: object, target_path: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("second replace denied")
        original_replace(source_path, target_path)

    monkeypatch.setattr(application_module.os, "replace", fail_second)
    with pytest.raises(PatchApplicationError, match="atomically replace"):
        _apply(patch_workspace, patch_task, reproduced_result, proposal)

    assert [path.read_bytes() for path in paths] == before
    assert calls == 3
    assert _temporary_paths(patch_workspace) == []


def test_rollback_failure_reports_possible_manual_restoration(
    monkeypatch: pytest.MonkeyPatch, patch_workspace, patch_task, reproduced_result
) -> None:
    reproduced_result, draft = _multi_file_setup(patch_workspace, reproduced_result)
    proposal = _proposal(patch_workspace, patch_task, reproduced_result, draft)
    original_replace = application_module.os.replace
    calls = 0

    def fail_second_and_rollback(source_path: object, target_path: object) -> None:
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise PermissionError("replace denied")
        original_replace(source_path, target_path)

    monkeypatch.setattr(application_module.os, "replace", fail_second_and_rollback)
    with pytest.raises(PatchApplicationError, match="manual restoration"):
        _apply(patch_workspace, patch_task, reproduced_result, proposal)

    assert _temporary_paths(patch_workspace) == []


def test_programmer_errors_are_not_normalized(
    monkeypatch: pytest.MonkeyPatch, patch_workspace, patch_task, reproduced_result
) -> None:
    proposal = _proposal(patch_workspace, patch_task, reproduced_result)

    def fail_programmer_contract(*args: object, **kwargs: object) -> object:
        raise TypeError("internal contract failure")

    monkeypatch.setattr(application_module, "_preflight_files", fail_programmer_contract)
    with pytest.raises(TypeError, match="internal contract failure"):
        _apply(patch_workspace, patch_task, reproduced_result, proposal)
