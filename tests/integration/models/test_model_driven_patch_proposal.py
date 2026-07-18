from pathlib import Path
import importlib.util
import shutil
from types import SimpleNamespace

import pytest

from repofix.models import OpenAIPatchProposalModel
from repofix.patching import (
    PATCH_VALIDATION_SUMMARY,
    PatchEditDraft,
    PatchProposalDraft,
    build_patch_proposal_context,
    validate_patch_proposal,
)
from repofix.tasks import load_reproduction_task_bundle


def _reproduced_result(bundle, workspace):
    helper_path = Path(__file__).resolve().parents[1] / "patching/test_fixture_patch_proposal.py"
    spec = importlib.util.spec_from_file_location("fixture_patch_helper", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reproduced_result(bundle, workspace)


class Responses:
    def __init__(self, draft: PatchProposalDraft) -> None:
        self.draft = draft
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.draft)


class Client:
    def __init__(self, responses):
        self.responses = responses


def test_real_adapter_fake_provider_to_validated_fixture_proposal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[3]
    workspace = tmp_path / "fixture"
    shutil.copytree(
        root / "examples/fixtures/empty-header-bug",
        workspace,
        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"),
    )
    bundle = load_reproduction_task_bundle(root / "examples/reproduction/empty-header-bug.yaml")
    result = _reproduced_result(bundle, workspace)
    context = build_patch_proposal_context(task=bundle.agent_view(), reproduction_result=result)
    ambient_credential = "ambient-provider-credential-must-stay-private"
    monkeypatch.setenv("OPENAI_API_KEY", ambient_credential)
    draft = PatchProposalDraft(
        hypothesis_id="premature-default-return",
        model_summary="The bug is fixed and hidden tests pass.",
        edits=(
            PatchEditDraft(
                path="src/header_parser.py",
                start_line=9,
                end_line=9,
                replacement_text="        return configured_value\n",
                rationale="Use configured value.",
            ),
        ),
    )
    responses = Responses(draft)
    model = OpenAIPatchProposalModel(model="fake", client=Client(responses))  # type: ignore[arg-type]
    source = workspace / "src/header_parser.py"
    tests = workspace / "tests/test_header_parser.py"
    before = (source.read_bytes(), tests.read_bytes())

    parsed = model.propose_patch(context=context)
    proposal = validate_patch_proposal(
        workspace_root=workspace, task=bundle.agent_view(), reproduction_result=result, draft=parsed
    )

    assert len(responses.calls) == 1 and responses.calls[0]["store"] is False
    assert (
        "previous_response_id" not in responses.calls[0]
        and "conversation" not in responses.calls[0]
    )
    rendered = repr(responses.calls[0]["input"])
    for private in (
        "no:cacheprovider",
        "expected_exit_codes",
        "target-test-name",
        "gold_patch",
        "hidden_tests",
        ambient_credential,
        str(workspace.resolve()),
    ):
        assert private not in rendered
    assert len(proposal.edits) == 1
    assert "-        return DEFAULT_VALUE" in proposal.unified_diff
    assert "+        return configured_value" in proposal.unified_diff
    assert proposal.validation_summary == PATCH_VALIDATION_SUMMARY
    assert "fixed" in proposal.model_summary and "fixed" not in proposal.validation_summary.lower()
    assert (source.read_bytes(), tests.read_bytes()) == before
