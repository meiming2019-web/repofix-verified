import pytest
from pydantic import ValidationError

from repofix.patching import (
    PatchEditDraft,
    PatchProposalDraft,
    ValidatedPatchProposal,
    validate_patch_proposal,
)
from repofix.patching.models import (
    MAX_PATCH_EDITS,
    MAX_REPLACEMENT_CHARS,
)


def edit(**updates: object) -> dict[str, object]:
    data: dict[str, object] = {
        "path": "src/app.py",
        "start_line": 2,
        "end_line": 2,
        "replacement_text": "    return 'right'\n",
        "rationale": "Use configured value.",
    }
    data.update(updates)
    return data


def test_valid_strict_frozen_draft() -> None:
    draft = PatchProposalDraft.model_validate(
        {"hypothesis_id": "h1", "model_summary": "Correct return.", "edits": [edit()]}
    )
    assert draft.edits[0].path == "src/app.py"
    with pytest.raises(ValidationError):
        draft.model_summary = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "updates",
    [
        {"path": "../x"},
        {"start_line": True},
        {"start_line": 3, "end_line": 2},
        {"replacement_text": "x\0"},
        {"rationale": ""},
        {"unknown": 1},
    ],
)
def test_invalid_edit_fields(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PatchEditDraft.model_validate(edit(**updates))


def test_draft_bounds() -> None:
    with pytest.raises(ValidationError):
        PatchProposalDraft(
            hypothesis_id="", model_summary="x", edits=(PatchEditDraft.model_validate(edit()),)
        )
    with pytest.raises(ValidationError):
        PatchProposalDraft.model_validate({"hypothesis_id": "h", "model_summary": "x", "edits": []})
    with pytest.raises(ValidationError):
        PatchEditDraft.model_validate(edit(replacement_text="x" * (MAX_REPLACEMENT_CHARS + 1)))
    edits = [edit(path=f"src/{i}.py", start_line=1, end_line=1) for i in range(MAX_PATCH_EDITS + 1)]
    with pytest.raises(ValidationError):
        PatchProposalDraft.model_validate(
            {"hypothesis_id": "h", "model_summary": "x", "edits": edits}
        )


def test_forged_diff_header_path_is_rejected() -> None:
    for separator in ("\n", "\u2028", "\u2029"):
        with pytest.raises(ValidationError, match="control"):
            PatchEditDraft.model_validate(edit(path=f"src/evil{separator}+++ b/forged.py"))


def valid_proposal_data(patch_workspace, patch_task, reproduced_result) -> dict[str, object]:
    draft = PatchProposalDraft.model_validate(
        {"hypothesis_id": "h1", "model_summary": "change", "edits": [edit()]}
    )
    proposal = validate_patch_proposal(
        workspace_root=patch_workspace,
        task=patch_task,
        reproduction_result=reproduced_result,
        draft=draft,
    )
    return proposal.model_dump()


def test_validated_model_rejects_edit_and_file_count_bounds(
    patch_workspace, patch_task, reproduced_result
) -> None:
    data = valid_proposal_data(patch_workspace, patch_task, reproduced_result)
    base_edit = data["edits"][0]  # type: ignore[index]
    nine_edits = [
        {**base_edit, "start_line": line, "end_line": line}  # type: ignore[arg-type]
        for line in range(1, MAX_PATCH_EDITS + 2)
    ]
    with pytest.raises(ValidationError, match="at most 8"):
        ValidatedPatchProposal.model_validate({**data, "edits": nine_edits})

    four_edits = []
    four_snapshots = []
    for index in range(4):
        path = f"src/{index}.py"
        four_edits.append({**base_edit, "path": path})  # type: ignore[arg-type]
        snapshot = data["file_snapshots"][0]  # type: ignore[index]
        four_snapshots.append({**snapshot, "path": path})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="at most 3|file limit"):
        ValidatedPatchProposal.model_validate(
            {**data, "edits": four_edits, "file_snapshots": four_snapshots}
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"model_summary": "   "}, "blank"),
        ({"file_snapshots": ()}, "at least 1|one snapshot"),
        ({"reproduction_expectation_fingerprint": "f" * 64}, "digest does not match"),
        ({"proposal_digest": "0" * 64}, "digest does not match"),
    ],
)
def test_validated_model_rejects_missing_or_altered_integrity_fields(
    patch_workspace, patch_task, reproduced_result, update, message
) -> None:
    data = valid_proposal_data(patch_workspace, patch_task, reproduced_result)

    with pytest.raises(ValidationError, match=message):
        ValidatedPatchProposal.model_validate({**data, **update})


def test_validated_model_rejects_noop_and_inconsistent_file_hashes(
    patch_workspace, patch_task, reproduced_result
) -> None:
    data = valid_proposal_data(patch_workspace, patch_task, reproduced_result)
    base_edit = data["edits"][0]  # type: ignore[index]
    noop = {**base_edit, "replacement_text": base_edit["original_text"]}  # type: ignore[arg-type,index]
    with pytest.raises(ValidationError, match="must change"):
        ValidatedPatchProposal.model_validate({**data, "edits": [noop]})

    inconsistent = [
        {**base_edit, "start_line": 1, "end_line": 1},  # type: ignore[arg-type]
        {
            **base_edit,  # type: ignore[arg-type]
            "start_line": 2,
            "end_line": 2,
            "original_file_sha256": "f" * 64,
        },
    ]
    with pytest.raises(ValidationError, match="share the source hash"):
        ValidatedPatchProposal.model_validate({**data, "edits": inconsistent})
