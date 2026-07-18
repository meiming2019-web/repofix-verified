import hashlib
import os

import pytest

import repofix.patching.validator as validator_module
from repofix.agent import ToolObservation
from repofix.patching import (
    PatchProposalDraft,
    PatchProposalValidationError,
    validate_patch_proposal,
)
from repofix.patching.models import MAX_REPLACEMENT_CHARS
from repofix.patching.models import MAX_TOTAL_REPLACEMENT_CHARS


def draft(path="src/app.py", replacement="    return 'right'\n"):
    return PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "h1",
            "model_summary": "Correct the return without claiming verification.",
            "edits": [
                {
                    "path": path,
                    "start_line": 2,
                    "end_line": 2,
                    "replacement_text": replacement,
                    "rationale": "Use the expected value.",
                }
            ],
        }
    )


def with_successful_read(reproduced_result, workspace, path: str):
    observation = ToolObservation(
        step_index=len(reproduced_result.state.observations) + 1,
        tool_name="read_file",
        arguments={"path": path, "start_line": 1, "end_line": 2},
        success=True,
        output="read",
        error=None,
        full_file_sha256=hashlib.sha256((workspace / path).read_bytes()).hexdigest(),
    )
    state = reproduced_result.state.model_copy(
        update={"observations": (*reproduced_result.state.observations, observation)}
    )
    return reproduced_result.model_copy(update={"state": state})


def with_current_read_hash(reproduced_result, workspace, path: str = "src/app.py"):
    observations = tuple(
        observation.model_copy(
            update={"full_file_sha256": hashlib.sha256((workspace / path).read_bytes()).hexdigest()}
        )
        if observation.tool_name == "read_file" and observation.arguments.get("path") == path
        else observation
        for observation in reproduced_result.state.observations
    )
    state = reproduced_result.state.model_copy(update={"observations": observations})
    return reproduced_result.model_copy(update={"state": state})


def test_valid_proposal_is_deterministic_and_does_not_write(
    patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    before = path.read_bytes()
    first = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=draft(),
    )
    second = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=draft(),
    )
    assert first == second
    assert first.edits[0].original_text == "    return 'wrong'\n"
    assert first.edits[0].original_file_sha256 == hashlib.sha256(before).hexdigest()
    assert (
        "-    return 'wrong'" in first.unified_diff and "+    return 'right'" in first.unified_diff
    )
    assert path.read_bytes() == before


def test_accepted_draft_resolves_to_exactly_one_supported_hypothesis(
    patch_workspace, patch_task, reproduced_result
) -> None:
    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=draft(),
    )

    assert proposal.hypothesis_id == "h1"
    assert [
        hypothesis.hypothesis_id
        for hypothesis in reproduced_result.state.hypotheses
        if hypothesis.status == "supported"
    ] == [proposal.hypothesis_id]


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("tests/test_app.py", "outside patchable"),
        ("src/missing.py", "not previously read"),
    ],
)
def test_rejects_unsafe_targets(
    patch_workspace, patch_task, reproduced_result, path, message
) -> None:
    with pytest.raises(PatchProposalValidationError, match=message):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(path),
        )


def test_rejects_unread_unchanged_and_out_of_range(
    patch_workspace, patch_task, reproduced_result
) -> None:
    for value in (
        draft(replacement="    return 'wrong'\n"),
        PatchProposalDraft.model_validate(
            {
                "hypothesis_id": "h1",
                "model_summary": "x",
                "edits": [
                    {
                        "path": "src/app.py",
                        "start_line": 9,
                        "end_line": 9,
                        "replacement_text": "x",
                        "rationale": "x",
                    }
                ],
            }
        ),
    ):
        with pytest.raises(PatchProposalValidationError):
            validate_patch_proposal(
                workspace_root=patch_workspace,
                task=patch_task,
                reproduction_result=reproduced_result,
                draft=value,
            )


def test_target_changed_after_reproduction_read_is_rejected(
    patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    path.write_text("def value():\n    return 'completely different'\n", encoding="utf-8")

    with pytest.raises(PatchProposalValidationError, match="reproduction read"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(),
        )


def test_conflicting_successful_read_hashes_are_rejected(
    patch_workspace, patch_task, reproduced_result
) -> None:
    conflicting = ToolObservation(
        step_index=2,
        tool_name="read_file",
        arguments={"path": "src/app.py", "start_line": 1, "end_line": 2},
        success=True,
        output="different read",
        error=None,
        full_file_sha256="f" * 64,
    )
    state = reproduced_result.state.model_copy(
        update={"observations": (*reproduced_result.state.observations, conflicting)}
    )
    result = reproduced_result.model_copy(update={"state": state})

    with pytest.raises(PatchProposalValidationError, match="conflicting full-file hashes"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=result,
            draft=draft(),
        )


def test_validator_defensively_requires_one_supported_hypothesis_identity(
    patch_workspace, patch_task, reproduced_result
) -> None:
    duplicate = reproduced_result.state.hypotheses[0].model_copy(
        update={"description": "different root cause"}
    )
    state = reproduced_result.state.model_copy(
        update={"hypotheses": (*reproduced_result.state.hypotheses, duplicate)}
    )
    result = reproduced_result.model_copy(update={"state": state})

    with pytest.raises(PatchProposalValidationError, match="duplicate supported"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=result,
            draft=draft(),
        )


def test_eof_newline_changes_are_rejected_but_no_newline_final_edit_is_diffed(
    patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    path.write_text("first\nlast", encoding="utf-8")
    reproduced_result = with_current_read_hash(reproduced_result, patch_workspace)
    final = PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "h1",
            "model_summary": "x",
            "edits": [
                {
                    "path": "src/app.py",
                    "start_line": 2,
                    "end_line": 2,
                    "replacement_text": "changed",
                    "rationale": "x",
                }
            ],
        }
    )
    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=final,
    )
    assert "-last" in proposal.unified_diff and "+changed" in proposal.unified_diff
    adding = final.model_copy(
        update={"edits": (final.edits[0].model_copy(update={"replacement_text": "changed\n"}),)}
    )
    with pytest.raises(PatchProposalValidationError, match="final newline"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=adding,
        )
    path.write_text("first\nlast\n", encoding="utf-8")
    reproduced_result = with_current_read_hash(reproduced_result, patch_workspace)
    removing = final.model_copy(
        update={"edits": (final.edits[0].model_copy(update={"replacement_text": "changed"}),)}
    )
    with pytest.raises(PatchProposalValidationError, match="final newline"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=removing,
        )


def test_nonfinal_replacement_without_newline_is_line_safe(
    patch_workspace, patch_task, reproduced_result
) -> None:
    value = PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "h1",
            "model_summary": "x",
            "edits": [
                {
                    "path": "src/app.py",
                    "start_line": 1,
                    "end_line": 1,
                    "replacement_text": "def corrected():",
                    "rationale": "x",
                }
            ],
        }
    )
    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=value,
    )
    assert proposal.edits[0].replacement_text.endswith("\n")
    assert "+def corrected():" in proposal.unified_diff


def test_logical_path_symlink_swap_is_detected(
    monkeypatch: pytest.MonkeyPatch, patch_workspace, patch_task, reproduced_result
) -> None:
    original_verify = validator_module._verify_snapshot
    logical = patch_workspace / "src/app.py"
    moved = patch_workspace / "src/original.py"
    external = patch_workspace.parent / f"{patch_workspace.name}-external.py"

    def swap_then_verify(*, workspace, snapshot):
        logical.rename(moved)
        external.write_bytes(moved.read_bytes())
        try:
            logical.symlink_to(external)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
        original_verify(workspace=workspace, snapshot=snapshot)

    monkeypatch.setattr(validator_module, "_verify_snapshot", swap_then_verify)
    with pytest.raises(PatchProposalValidationError, match="identity changed"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(),
        )


def test_validated_proposal_digest_is_self_checked(
    patch_workspace, patch_task, reproduced_result
) -> None:
    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=draft(),
    )
    with pytest.raises(ValueError, match="digest does not match"):
        proposal.model_validate({**proposal.model_dump(), "proposal_digest": "0" * 64})
    with pytest.raises(ValueError, match="digest does not match"):
        proposal.model_validate({**proposal.model_dump(), "task_fingerprint": "1" * 64})


def test_crlf_replacement_is_normalized_and_mixed_sources_are_rejected(
    patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    path.write_bytes(b"def value():\r\n    return 'wrong'\r\n")
    reproduced_result = with_current_read_hash(reproduced_result, patch_workspace)
    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=draft(replacement="    return 'right'\n"),
    )
    assert proposal.edits[0].original_text.endswith("\r\n")
    assert proposal.edits[0].replacement_text.endswith("\r\n")
    assert proposal.edits[0].replacement_text != "    return 'right'\n"
    path.write_bytes(b"def value():\r\n    return 'wrong'\n")
    reproduced_result = with_current_read_hash(reproduced_result, patch_workspace)
    with pytest.raises(PatchProposalValidationError, match="mixed line endings"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(),
        )


def test_mutation_during_digest_computation_is_caught_by_final_snapshot_check(
    monkeypatch: pytest.MonkeyPatch, patch_workspace, patch_task, reproduced_result
) -> None:
    original = validator_module.compute_proposal_digest
    path = patch_workspace / "src/app.py"

    def mutate_then_digest(**kwargs):
        value = original(**kwargs)
        path.write_text("mutated\n", encoding="utf-8")
        return value

    monkeypatch.setattr(validator_module, "compute_proposal_digest", mutate_then_digest)
    with pytest.raises(PatchProposalValidationError, match="identity changed"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(),
        )


def test_symlink_swap_inside_final_descriptor_read_is_rejected(
    monkeypatch: pytest.MonkeyPatch, patch_workspace, patch_task, reproduced_result
) -> None:
    logical = patch_workspace / "src/app.py"
    moved = patch_workspace / "src/original.py"
    external = patch_workspace.parent / f"{patch_workspace.name}-read-swap.py"
    original_read = validator_module.os.read
    original_verify = validator_module._verify_snapshot
    swapped = False
    final_verification = False

    def mark_final_verification(*, workspace, snapshot):
        nonlocal final_verification
        final_verification = True
        return original_verify(workspace=workspace, snapshot=snapshot)

    def swap_during_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if final_verification and not swapped:
            swapped = True
            logical.rename(moved)
            external.write_bytes(moved.read_bytes())
            try:
                logical.symlink_to(external)
            except OSError as error:
                pytest.skip(f"symlinks unavailable: {error}")
        return original_read(descriptor, size)

    monkeypatch.setattr(validator_module.os, "read", swap_during_read)
    monkeypatch.setattr(validator_module, "_verify_snapshot", mark_final_verification)
    with pytest.raises(PatchProposalValidationError, match="identity changed"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(),
        )
    assert swapped is True


def test_same_length_in_place_mutation_after_descriptor_read_is_rejected(
    monkeypatch: pytest.MonkeyPatch, patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    original_exact_check = validator_module._require_exact_logical_spelling
    checks = 0

    def mutate_before_post_read_logical_check(workspace, logical):
        nonlocal checks
        checks += 1
        if checks == 3:
            before = path.stat()
            path.write_bytes(b"def value():\n    return 'other'\n")
            after = path.stat()
            assert (before.st_dev, before.st_ino, before.st_mode, before.st_size) == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
            )
        original_exact_check(workspace, logical)

    monkeypatch.setattr(
        validator_module,
        "_require_exact_logical_spelling",
        mutate_before_post_read_logical_check,
    )

    with pytest.raises(PatchProposalValidationError, match="identity changed"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(),
        )


@pytest.mark.parametrize("mutation", ["growth", "mode", "hard_link"])
def test_post_read_metadata_mutations_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    patch_workspace,
    patch_task,
    reproduced_result,
    mutation: str,
) -> None:
    path = patch_workspace / "src/app.py"
    outside_alias = patch_workspace.parent / f"{patch_workspace.name}-outside-alias.py"
    original_exact_check = validator_module._require_exact_logical_spelling
    checks = 0

    def mutate(workspace, logical):
        nonlocal checks
        checks += 1
        if checks == 3:
            if mutation == "growth":
                with path.open("ab") as stream:
                    stream.write(b"growth\n")
            elif mutation == "mode":
                path.chmod(0o600)
            else:
                try:
                    os.link(path, outside_alias)
                except OSError as error:
                    pytest.skip(f"hard links unavailable: {error}")
        original_exact_check(workspace, logical)

    monkeypatch.setattr(validator_module, "_require_exact_logical_spelling", mutate)

    with pytest.raises(PatchProposalValidationError):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(),
        )


def test_exact_spelling_rename_during_final_verification_is_rejected(
    monkeypatch: pytest.MonkeyPatch, patch_workspace, patch_task, reproduced_result
) -> None:
    logical = patch_workspace / "src/app.py"
    renamed = patch_workspace / "src/App.py"
    original_exact_check = validator_module._require_exact_logical_spelling
    original_verify = validator_module._verify_snapshot
    final_verification = False
    final_checks = 0

    def mark_final(*, workspace, snapshot):
        nonlocal final_verification
        final_verification = True
        return original_verify(workspace=workspace, snapshot=snapshot)

    def rename_before_final_post_check(workspace, logical_path):
        nonlocal final_checks
        if final_verification:
            final_checks += 1
            if final_checks == 2:
                logical.rename(renamed)
        original_exact_check(workspace, logical_path)

    monkeypatch.setattr(validator_module, "_verify_snapshot", mark_final)
    monkeypatch.setattr(
        validator_module, "_require_exact_logical_spelling", rename_before_final_post_check
    )

    with pytest.raises(PatchProposalValidationError, match="identity changed"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(),
        )


def test_sparse_oversized_file_is_rejected_without_unbounded_path_read(
    monkeypatch: pytest.MonkeyPatch, patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    with path.open("wb") as stream:
        stream.truncate(validator_module.MAX_PATCH_SOURCE_FILE_BYTES + 1)

    def fail_unbounded_read(self):
        raise AssertionError("Path.read_bytes must not be used by patch snapshots")

    monkeypatch.setattr(validator_module.Path, "read_bytes", fail_unbounded_read)

    with pytest.raises(PatchProposalValidationError, match="byte limit"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(),
        )


def test_single_target_with_external_hard_link_is_rejected(
    patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    outside = patch_workspace.parent / f"{patch_workspace.name}-external-hard-link.py"
    try:
        os.link(path, outside)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")
    before = path.read_bytes()

    with pytest.raises(PatchProposalValidationError, match="exactly one hard link"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(),
        )

    assert path.read_bytes() == before


def test_unicode_line_separators_cannot_enter_generated_diff() -> None:
    for separator in ("\u2028", "\u2029"):
        with pytest.raises(ValueError, match="control"):
            draft(path=f"src/evil{separator}+++ b/forged.py")


def test_one_deletion_cannot_empty_file_and_source_remains_unchanged(
    patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    path.write_text("only line\n", encoding="utf-8")
    reproduced_result = with_current_read_hash(reproduced_result, patch_workspace)
    before = path.read_bytes()
    whole_file_deletion = PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "h1",
            "model_summary": "delete all",
            "edits": [
                {
                    "path": "src/app.py",
                    "start_line": 1,
                    "end_line": 1,
                    "replacement_text": "",
                    "rationale": "x",
                }
            ],
        }
    )
    with pytest.raises(PatchProposalValidationError, match="empty an entire file"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=whole_file_deletion,
        )
    assert path.read_bytes() == before


def test_combined_deletions_cannot_empty_file_and_source_remains_unchanged(
    patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    path.write_text("first\nsecond", encoding="utf-8")
    reproduced_result = with_current_read_hash(reproduced_result, patch_workspace)
    before = path.read_bytes()
    two_deletions = PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "h1",
            "model_summary": "delete all",
            "edits": [
                {
                    "path": "src/app.py",
                    "start_line": line,
                    "end_line": line,
                    "replacement_text": "",
                    "rationale": "x",
                }
                for line in (1, 2)
            ],
        }
    )
    with pytest.raises(PatchProposalValidationError, match="empty an entire file"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=two_deletions,
        )
    assert path.read_bytes() == before


def test_multiple_deletions_that_leave_content_are_valid(
    patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    path.write_text("first\nsecond\nthird", encoding="utf-8")
    reproduced_result = with_current_read_hash(reproduced_result, patch_workspace)
    before = path.read_bytes()
    partial = PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "h1",
            "model_summary": "delete some",
            "edits": [
                {
                    "path": "src/app.py",
                    "start_line": line,
                    "end_line": line,
                    "replacement_text": "",
                    "rationale": "x",
                }
                for line in (1, 2)
            ],
        }
    )
    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=partial,
    )
    assert all(edit.replacement_text == "" for edit in proposal.edits)
    assert "+third" not in proposal.unified_diff
    assert path.read_bytes() == before


def test_lf_replacement_at_limit_is_valid_and_validated_lengths_are_bounded(
    patch_workspace, patch_task, reproduced_result
) -> None:
    replacement = "x" * (MAX_REPLACEMENT_CHARS - 1) + "\n"

    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=draft(replacement=replacement),
    )

    assert len(proposal.edits[0].replacement_text) == MAX_REPLACEMENT_CHARS
    assert all(len(edit.replacement_text) <= MAX_REPLACEMENT_CHARS for edit in proposal.edits)
    assert sum(len(edit.replacement_text) for edit in proposal.edits) <= MAX_TOTAL_REPLACEMENT_CHARS


def test_lf_replacement_above_limit_is_rejected_by_draft_model() -> None:
    with pytest.raises(ValueError, match="at most"):
        draft(replacement="x" * (MAX_REPLACEMENT_CHARS + 1))


def test_crlf_expansion_beyond_per_edit_limit_is_rejected(
    patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    path.write_bytes(b"one\r\ntwo\r\nthree\r\nfour\r\n")
    reproduced_result = with_current_read_hash(reproduced_result, patch_workspace)
    at_limit = "x" * (MAX_REPLACEMENT_CHARS - 1) + "\n"
    with pytest.raises(PatchProposalValidationError, match="character limit"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=draft(replacement=at_limit),
        )


def test_crlf_expansion_beyond_effective_total_limit_is_rejected(
    patch_workspace, patch_task, reproduced_result
) -> None:
    path = patch_workspace / "src/app.py"
    path.write_bytes(b"one\r\ntwo\r\nthree\r\nfour\r\n")
    reproduced_result = with_current_read_hash(reproduced_result, patch_workspace)
    expanded_total = PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "h1",
            "model_summary": "large",
            "edits": [
                {
                    "path": "src/app.py",
                    "start_line": line,
                    "end_line": line,
                    "replacement_text": "x\n" * 3000,
                    "rationale": "x",
                }
                for line in (1, 2, 3)
            ],
        }
    )
    with pytest.raises(PatchProposalValidationError, match="total character limit"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=reproduced_result,
            draft=expanded_total,
        )


def test_same_logical_path_with_nonoverlapping_edits_remains_valid(
    patch_workspace, patch_task, reproduced_result
) -> None:
    value = PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "h1",
            "model_summary": "two edits",
            "edits": [
                {
                    "path": "src/app.py",
                    "start_line": 1,
                    "end_line": 1,
                    "replacement_text": "def corrected():\n",
                    "rationale": "x",
                },
                {
                    "path": "src/app.py",
                    "start_line": 2,
                    "end_line": 2,
                    "replacement_text": "    return 'right'\n",
                    "rationale": "x",
                },
            ],
        }
    )

    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=value,
    )

    assert len(proposal.edits) == 2
    assert {edit.path for edit in proposal.edits} == {"src/app.py"}


def test_distinct_hard_link_paths_are_rejected(
    patch_workspace, patch_task, reproduced_result
) -> None:
    alias = patch_workspace / "src/alias.py"
    try:
        os.link(patch_workspace / "src/app.py", alias)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")
    result = with_successful_read(reproduced_result, patch_workspace, "src/alias.py")
    value = PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "h1",
            "model_summary": "aliases",
            "edits": [
                {
                    "path": "src/alias.py",
                    "start_line": 1,
                    "end_line": 1,
                    "replacement_text": "alias\n",
                    "rationale": "x",
                },
                {
                    "path": "src/app.py",
                    "start_line": 2,
                    "end_line": 2,
                    "replacement_text": "app\n",
                    "rationale": "x",
                },
            ],
        }
    )
    before = (alias.read_bytes(), (patch_workspace / "src/app.py").read_bytes())
    with pytest.raises(PatchProposalValidationError, match="same physical file"):
        validate_patch_proposal(
            workspace_root=patch_workspace, task=patch_task, reproduction_result=result, draft=value
        )
    assert (alias.read_bytes(), (patch_workspace / "src/app.py").read_bytes()) == before


def test_case_alias_is_rejected_on_case_insensitive_filesystems(
    patch_workspace, patch_task, reproduced_result
) -> None:
    exact = patch_workspace / "src/CaseFile.py"
    alias = patch_workspace / "src/casefile.py"
    exact.write_text("wrong\n", encoding="utf-8")
    if not alias.exists():
        pytest.skip("filesystem is case-sensitive")
    result = with_successful_read(reproduced_result, patch_workspace, "src/casefile.py")
    before = exact.read_bytes()

    with pytest.raises(PatchProposalValidationError, match="exact repository spelling"):
        validate_patch_proposal(
            workspace_root=patch_workspace,
            task=patch_task,
            reproduction_result=result,
            draft=draft(path="src/casefile.py", replacement="right\n"),
        )

    assert exact.read_bytes() == before


def test_exact_ordinary_unicode_path_is_accepted(
    patch_workspace, patch_task, reproduced_result
) -> None:
    relative = "src/源.py"
    path = patch_workspace / "src/源.py"
    path.write_text("def value():\n    return 'wrong'\n", encoding="utf-8")
    result = with_successful_read(reproduced_result, patch_workspace, relative)

    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=result,
        draft=draft(path=relative, replacement="right\n"),
    )

    assert proposal.edits[0].path == relative


def test_validated_proposal_contains_unique_physical_file_identities(
    patch_workspace, patch_task, reproduced_result
) -> None:
    other_relative = "src/other.py"
    (patch_workspace / other_relative).write_text("wrong\n", encoding="utf-8")
    result = with_successful_read(reproduced_result, patch_workspace, other_relative)
    value = PatchProposalDraft.model_validate(
        {
            "hypothesis_id": "h1",
            "model_summary": "two independent files",
            "edits": [
                {
                    "path": "src/app.py",
                    "start_line": 2,
                    "end_line": 2,
                    "replacement_text": "    return 'right'\n",
                    "rationale": "x",
                },
                {
                    "path": other_relative,
                    "start_line": 1,
                    "end_line": 1,
                    "replacement_text": "right\n",
                    "rationale": "x",
                },
            ],
        }
    )

    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=result,
        draft=value,
    )
    identities = {
        ((patch_workspace / edit.path).stat().st_dev, (patch_workspace / edit.path).stat().st_ino)
        for edit in proposal.edits
    }

    assert len(identities) == len({edit.path for edit in proposal.edits})
