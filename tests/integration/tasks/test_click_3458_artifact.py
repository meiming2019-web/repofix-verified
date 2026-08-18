"""Structural validation for the formal Click #3458 evaluator artifact."""

import hashlib
from pathlib import Path

from repofix.agent.reproduction_loop import compute_task_fingerprint
from repofix.run_artifacts import RunIdentity
from repofix.tasks import (
    AgentTaskSpec,
    EvaluatorTaskBundle,
    load_agent_task_spec,
    load_evaluator_task_bundle,
)


REPOSITORY_ROOT = Path(__file__).parents[3]
ARTIFACT_ROOT = REPOSITORY_ROOT / "examples/evaluator/click-3458"
TASK_PATH = ARTIFACT_ROOT / "task.yaml"


def test_click_3458_evaluator_artifact_is_canonical_and_agent_safe() -> None:
    bundle = load_evaluator_task_bundle(TASK_PATH)
    task = bundle.agent_view()
    agent_loaded = load_agent_task_spec(TASK_PATH)
    hidden_reference = bundle.hidden_verification.test_file
    hidden_path = ARTIFACT_ROOT / hidden_reference.path
    hidden_contents = hidden_path.read_bytes()

    assert type(bundle) is EvaluatorTaskBundle
    assert type(task) is AgentTaskSpec
    assert agent_loaded == task
    assert task.task_id == "click-3458-parameter-source-during-conversion"
    assert task.repository_url == "https://github.com/pallets/click.git"
    assert task.pre_fix_commit == "4b24a6c8f6658ef89fd6db4dcaf8ee88eef75dfa"
    assert task.allowed_source_paths == ("src/click", "tests")
    assert task.patchable_source_paths == ("src/click",)
    task_fingerprint = compute_task_fingerprint(task)
    assert task_fingerprint == compute_task_fingerprint(agent_loaded)
    assert (
        task_fingerprint
        == "f6d6c8cc9728e994a89fa94e675317a93731158f481bfd800b9c6137f0edb8b0"
    )

    reproduction = task.approved_commands["reproduce-click-3458"]
    regression = task.approved_commands["regression-parameter-source"]
    assert reproduction.argv[:3] == ("python", "-B", "-c")
    assert "REPOFIX_CLICK_3458_REPRODUCED" in reproduction.argv[3]
    assert "REPOFIX_CLICK_3458_NOT_REPRODUCED" in reproduction.argv[3]
    assert regression.argv[:3] == ("python", "-B", "-c")
    assert "tests/test_context.py::test_parameter_source" in regression.argv[3]
    assert bundle.reproduction.command_id == "reproduce-click-3458"
    assert bundle.reproduction.expected_exit_codes == (1,)
    assert tuple(
        (fragment.stream.value, fragment.text)
        for fragment in bundle.reproduction.required_fragments
    ) == (("stdout", "REPOFIX_CLICK_3458_REPRODUCED"),)
    assert bundle.reproduction.forbidden_fragments == ()
    assert bundle.regression.command_id == "regression-parameter-source"

    assert bundle.hidden_verification.command_id == "hidden-click-3458"
    assert bundle.hidden_verification.launcher.argv == ("python", "-B")
    assert hidden_reference.path == (
        "hidden_tests/click_3458_parameter_source_contract.py"
    )
    assert hidden_reference.size_bytes == len(hidden_contents)
    assert hidden_reference.sha256 == hashlib.sha256(hidden_contents).hexdigest()

    protected = {
        item.path: (item.sha256, item.size_bytes)
        for item in bundle.patch_policy.protected_files
    }
    assert protected == {
        "pyproject.toml": (
            "f9a2bc31f584a32f88b9d9bcaf7c34bf8ab714eaf124b01b714b0394ca0f327b",
            5516,
        ),
        "tests/conftest.py": (
            "192be82444d6048fd222e2d73cd5c3bec768d65c6e10c0ffc230f307f363590b",
            131,
        ),
        "tests/test_context.py": (
            "187b7b8f9b1e666af21629c93d69702239b06a8ecd4840b83ede37e7cc32adcf",
            23042,
        ),
    }
    assert bundle.patch_policy.forbidden_paths == (
        "conftest.py",
        "pytest.ini",
        "setup.cfg",
        "sitecustomize.py",
        "tests/pytest.ini",
        "tests/setup.cfg",
        "tests/sitecustomize.py",
        "tests/tox.ini",
        "tests/usercustomize.py",
        "tox.ini",
        "usercustomize.py",
    )

    gold = bundle.gold_patch.patch
    assert gold.count("diff --git ") == 1
    assert "diff --git a/src/click/core.py b/src/click/core.py" in gold
    assert "tests/" not in gold
    normalized_gold = "\n".join(line.rstrip() for line in gold.splitlines()).rstrip()
    assert hashlib.sha256(normalized_gold.encode()).hexdigest() == (
        "622545a6ab2a483760b606971cc23bd1b545ff507d92d1bedac5768337d1a01d"
    )

    agent_serialized = task.model_dump_json()
    assert set(task.model_dump()) == set(AgentTaskSpec.model_fields)
    corrected_commit = "7d05a59b9d46a415d85937630d1d812cf477f60a"
    for evaluator_only_value in (
        "hidden_verification",
        "hidden-click-3458",
        hidden_reference.path,
        hidden_reference.sha256,
        "patch_policy",
        "gold_patch",
        corrected_commit,
        "set_parameter_source",
        "consume_value",
        "process_value",
        "feature-switch",
        "restore the previous source",
    ):
        assert evaluator_only_value not in agent_serialized

    for host_path_fragment in (
        "/" + "Users" + "/",
        "/private" + "/tmp",
        "click-3458" + "/bin",
    ):
        assert host_path_fragment not in TASK_PATH.read_text(encoding="utf-8")
        assert host_path_fragment not in hidden_contents.decode("utf-8")

    run_identity = RunIdentity(
        run_id="click-3458-future-run",
        task_id=task.task_id,
        task_fingerprint=task_fingerprint,
        repository_url=task.repository_url,
        pre_fix_commit=task.pre_fix_commit,
        repofix_commit="a" * 40,
        model_provider="future-provider",
        model_identifier="future-model",
        agent_workflow="reproduction",
    )
    assert run_identity.task_fingerprint == task_fingerprint
    assert run_identity.timing is None
