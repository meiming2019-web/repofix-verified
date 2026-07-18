import pytest

from repofix.patching import PatchProposalContextError, build_patch_proposal_context


def test_context_is_sanitized_deterministic_and_nonmutating(patch_task, reproduced_result) -> None:
    before = reproduced_result.model_dump()
    first = build_patch_proposal_context(task=patch_task, reproduction_result=reproduced_result)
    second = build_patch_proposal_context(task=patch_task, reproduction_result=reproduced_result)
    assert first == second
    assert reproduced_result.model_dump() == before
    assert first.patchable_source_paths == ("src",)
    assert first.successful_file_observations[0].path == "src/app.py"
    rendered = repr(first.model_dump())
    for secret in (
        "argv",
        "expected_exit_codes",
        "matched_required_fragment_ids",
        "hidden_tests",
        "gold_patch",
    ):
        assert secret not in rendered


@pytest.mark.parametrize(
    "updates",
    [
        {"repository_url": "https://github.com/example/other.git"},
        {"pre_fix_commit": "1" * 40},
        {"issue_body": "different"},
        {"approved_commands": {}},
        {"patchable_source_paths": ("tests",)},
    ],
)
def test_task_fingerprint_mismatch_is_rejected(patch_task, reproduced_result, updates) -> None:
    changed = patch_task.model_copy(update=updates)
    with pytest.raises(PatchProposalContextError, match="fingerprint"):
        build_patch_proposal_context(task=changed, reproduction_result=reproduced_result)


def test_context_defensively_rejects_duplicate_supported_hypothesis_ids(
    patch_task, reproduced_result
) -> None:
    duplicate = reproduced_result.state.hypotheses[0].model_copy(
        update={"description": "different cause"}
    )
    state = reproduced_result.state.model_copy(
        update={"hypotheses": (*reproduced_result.state.hypotheses, duplicate)}
    )
    result = reproduced_result.model_copy(update={"state": state})

    with pytest.raises(PatchProposalContextError, match="duplicate supported"):
        build_patch_proposal_context(task=patch_task, reproduction_result=result)
