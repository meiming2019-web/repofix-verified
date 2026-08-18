"""Validate the sanitized public record for the first real attrs run."""

from pathlib import Path

from repofix.run_artifacts import PublicRunArtifact, RunOutcome, RunStage


REPOSITORY_ROOT = Path(__file__).parents[3]
RUN_PATH = REPOSITORY_ROOT / "examples/run_artifacts/attrs-670-run-001.json"


def test_attrs_670_public_run_artifact_is_sanitized_and_canonical() -> None:
    contents = RUN_PATH.read_text(encoding="utf-8")
    artifact = PublicRunArtifact.model_validate_json(contents)

    assert artifact.identity.run_id == "attrs-670-run-001"
    assert artifact.identity.task_id == "attrs-670-custom-eq-autodetect"
    assert artifact.identity.repofix_commit == (
        "cb9d218264d829ffc81eca5123d6ee7e12a723f4"
    )
    assert artifact.outcome is RunOutcome.EVALUATOR_PASSED
    assert tuple(record.stage for record in artifact.stages) == tuple(RunStage)
    assert artifact.action_counts.repository_inspection_calls == 4
    assert artifact.action_counts.approved_command_requests == 1
    assert artifact.action_counts.total_agent_actions == 7
    assert artifact.candidate is not None
    assert artifact.candidate.candidate_patch is not None
    assert artifact.candidate.patch_omitted_for_safety is False
    assert artifact.stages[-2].finding_count == 0
    fingerprints = {
        record.stage: record.result_fingerprint for record in artifact.stages
    }
    assert fingerprints == {
        RunStage.REPRODUCTION: (
            "2571d1308e00df1e351ffb6ae48d9bba11d157465cd067e1be22d893cd542aec"
        ),
        RunStage.PROPOSAL: None,
        RunStage.REGRESSION_BASELINE: (
            "5711c303d74026c1652beab9176ab1324f1328fefa95ed61c9dac14e40da4bf8"
        ),
        RunStage.APPLICATION: (
            "d516bfe47f8fb8d9e3d3171467e6ec7e614a3f4a9c57d68af1a388a9c3963108"
        ),
        RunStage.POST_PATCH_REPRODUCTION: (
            "68e43f643d4ae25f57add7292f38766fb6c6e6338466b6bfb44b5d58e8929ee2"
        ),
        RunStage.REGRESSION: (
            "1a2aa77a9778d6eecb58b390f18df8dc678973087717b15213bf33a46129caf4"
        ),
        RunStage.HIDDEN: (
            "78c6c486951cfb8a881d255b51687bac3b6f402619b4f69aaefc33a9f287c956"
        ),
        RunStage.POLICY: (
            "2bb70329169107f956deecd4b8a53cf4e778f938deedf02d64cc879e54bfa5e6"
        ),
        RunStage.FINAL_EVALUATION: None,
    }

    forbidden_fragments = (
        "/" + "Users" + "/",
        "/private" + "/tmp/",
        "repofix" + "-pilots",
        "OPENAI" + "_API_KEY",
        "Authorization" + ":",
        "hidden" + "_tests",
        "Gold" + "PatchSpec",
        "gold" + "_patch",
        "response" + "_id",
    )
    for fragment in forbidden_fragments:
        assert fragment not in contents
