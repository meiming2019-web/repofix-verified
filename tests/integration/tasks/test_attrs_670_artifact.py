"""Structural validation for the formal attrs #670 evaluator artifact."""

import hashlib
from pathlib import Path

from repofix.agent.reproduction_loop import compute_task_fingerprint
from repofix.tasks import (
    AgentTaskSpec,
    EvaluatorTaskBundle,
    load_agent_task_spec,
    load_evaluator_task_bundle,
)


REPOSITORY_ROOT = Path(__file__).parents[3]
ARTIFACT_ROOT = REPOSITORY_ROOT / "examples/evaluator/attrs-670"
TASK_PATH = ARTIFACT_ROOT / "task.yaml"


def test_attrs_670_evaluator_artifact_is_canonical_and_agent_safe() -> None:
    bundle = load_evaluator_task_bundle(TASK_PATH)
    task = bundle.agent_view()
    agent_loaded = load_agent_task_spec(TASK_PATH)
    hidden_reference = bundle.hidden_verification.test_file
    hidden_path = ARTIFACT_ROOT / hidden_reference.path
    hidden_contents = hidden_path.read_bytes()

    assert type(bundle) is EvaluatorTaskBundle
    assert type(task) is AgentTaskSpec
    assert agent_loaded == task
    assert task.task_id == "attrs-670-custom-eq-autodetect"
    assert task.repository_url == "https://github.com/python-attrs/attrs.git"
    assert task.pre_fix_commit == "1a29941d5cf74ca585780495cb6c17fd013ec861"
    assert task.allowed_source_paths == ("src/attr", "tests")
    assert task.patchable_source_paths == ("src/attr",)
    assert compute_task_fingerprint(task) == compute_task_fingerprint(agent_loaded)

    reproduction = task.approved_commands["reproduce-attrs-670"]
    regression = task.approved_commands["regression-next-gen-equality"]
    assert reproduction.argv[:3] == ("python", "-B", "-c")
    assert "REPOFIX_ATTRS_670_REPRODUCED" in reproduction.argv[3]
    assert "REPOFIX_ATTRS_670_NOT_REPRODUCED" in reproduction.argv[3]
    assert "eq=None" not in reproduction.argv[3]
    assert regression.argv[:3] == ("python", "-B", "-c")
    assert "tests/test_next_gen.py::TestNextGen::test_auto_attribs_detect" in (
        regression.argv[3]
    )

    assert bundle.reproduction.command_id == "reproduce-attrs-670"
    assert bundle.reproduction.expected_exit_codes == (1,)
    assert tuple(
        (fragment.stream.value, fragment.text)
        for fragment in bundle.reproduction.required_fragments
    ) == (("stdout", "REPOFIX_ATTRS_670_REPRODUCED"),)
    assert bundle.reproduction.forbidden_fragments == ()
    assert bundle.regression.command_id == "regression-next-gen-equality"

    assert bundle.hidden_verification.command_id == "hidden-attrs-670"
    assert bundle.hidden_verification.launcher.argv == ("python", "-B")
    assert hidden_reference.path == "hidden_tests/attrs_670_eq_contract.py"
    assert hidden_reference.size_bytes == len(hidden_contents)
    assert hidden_reference.sha256 == hashlib.sha256(hidden_contents).hexdigest()

    protected = {
        item.path: (item.sha256, item.size_bytes)
        for item in bundle.patch_policy.protected_files
    }
    assert protected == {
        "conftest.py": (
            "6e5426484ba175e1e7b3fe3ac275c1db5de96ef3344daa77043537f60eeee36f",
            615,
        ),
        "tests/test_next_gen.py": (
            "6b87e369fb90dfa3e7cc159b05e119ecb7789a17bdf93910d046d73fa8c1da84",
            2816,
        ),
        "tox.ini": (
            "e441a6db42b9dce8b77bdb3f90ccf1a5db551a0d00e91fe391ada77d27f06a2b",
            2988,
        ),
    }
    assert bundle.patch_policy.forbidden_paths == (
        "pytest.ini",
        "setup.cfg",
        "sitecustomize.py",
        "tests/conftest.py",
        "usercustomize.py",
    )

    gold = bundle.gold_patch.patch
    assert gold.count("diff --git ") == 1
    assert "diff --git a/src/attr/_next_gen.py b/src/attr/_next_gen.py" in gold
    assert "tests/test_next_gen.py" not in gold

    agent_serialized = task.model_dump_json()
    assert set(task.model_dump()) == set(AgentTaskSpec.model_fields)
    for evaluator_only_value in (
        "hidden_verification",
        "hidden-attrs-670",
        hidden_reference.path,
        hidden_reference.sha256,
        "patch_policy",
        "gold_patch",
        "eq=None",
    ):
        assert evaluator_only_value not in agent_serialized
    host_path_fragments = (
        "/" + "Users" + "/",
        "/private" + "/tmp",
        "attrs-670" + "/bin",
    )
    for host_path_fragment in host_path_fragments:
        assert host_path_fragment not in TASK_PATH.read_text(encoding="utf-8")
        assert host_path_fragment not in hidden_contents.decode("utf-8")
