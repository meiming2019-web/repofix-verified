"""Tests for strict evaluator-only reproduction models."""

from typing import Any

import pytest
from pydantic import ValidationError

from repofix.execution import (
    ApprovedCommandExecutionResult,
    CommandTerminationReason,
)
from repofix.reproduction import (
    MAX_REPRODUCTION_FRAGMENT_LENGTH,
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionOutputFragment,
    ReproductionOutputStream,
    ReproductionStatus,
    ReproductionTaskBundle,
    ReproductionVerdict,
    compute_reproduction_expectation_fingerprint,
)
from repofix.tasks import AgentTaskSpec


def task_data() -> dict[str, Any]:
    return {
        "task_id": "model-task",
        "repository_url": "https://github.com/example/project.git",
        "pre_fix_commit": "0123456789abcdef0123456789abcdef01234567",
        "issue_title": "Target behavior fails",
        "issue_body": "The target behavior produces an incorrect result.",
        "approved_commands": {"unit_tests": {"argv": ["pytest", "-q"]}},
        "allowed_source_paths": ["src", "tests"],
        "timeout_seconds": 300,
    }


def fragment(
    fragment_id: str = "target-failure", text: str = "target assertion"
) -> ReproductionOutputFragment:
    return ReproductionOutputFragment(
        fragment_id=fragment_id,
        stream=ReproductionOutputStream.COMBINED,
        text=text,
    )


def expectation_data() -> dict[str, object]:
    return {
        "command_id": "unit_tests",
        "expected_exit_codes": [1],
        "required_fragments": [
            {"fragment_id": "target-failure", "stream": "combined", "text": "target"}
        ],
        "forbidden_fragments": [
            {"fragment_id": "import-error", "stream": "combined", "text": "ImportError"}
        ],
    }


def execution_result() -> ApprovedCommandExecutionResult:
    return ApprovedCommandExecutionResult(
        command_id="unit_tests",
        argv=("pytest", "-q"),
        termination_reason=CommandTerminationReason.COMPLETED,
        exit_code=1,
        stdout="target assertion\n",
        stderr="",
        stdout_bytes=17,
        stderr_bytes=0,
        had_decode_errors=False,
    )


def test_valid_output_fragment() -> None:
    value = fragment()

    assert value.fragment_id == "target-failure"
    assert value.stream is ReproductionOutputStream.COMBINED


@pytest.mark.parametrize("fragment_id", ["", "-target", "Target", "target failure", "a.b"])
def test_rejects_invalid_fragment_id(fragment_id: str) -> None:
    with pytest.raises(ValidationError):
        fragment(fragment_id=fragment_id)


@pytest.mark.parametrize("text", ["", "   "])
def test_rejects_empty_fragment_text(text: str) -> None:
    with pytest.raises(ValidationError):
        fragment(text=text)


def test_rejects_nul_and_excessive_fragment_text() -> None:
    with pytest.raises(ValidationError):
        fragment(text="target\0failure")
    with pytest.raises(ValidationError):
        fragment(text="x" * (MAX_REPRODUCTION_FRAGMENT_LENGTH + 1))


def test_valid_expectation_accepts_yaml_lists_and_stores_tuples() -> None:
    value = ReproductionExpectation.model_validate(expectation_data())

    assert value.expected_exit_codes == (1,)
    assert isinstance(value.required_fragments, tuple)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_exit_codes", [2]),
        (
            "required_fragments",
            [{"fragment_id": "target-failure", "stream": "combined", "text": "changed"}],
        ),
        (
            "forbidden_fragments",
            [{"fragment_id": "import-error", "stream": "combined", "text": "changed"}],
        ),
        (
            "required_fragments",
            [{"fragment_id": "target-failure", "stream": "stderr", "text": "target"}],
        ),
    ],
)
def test_expectation_fingerprint_binds_every_evaluator_field(field: str, value: object) -> None:
    original = ReproductionExpectation.model_validate(expectation_data())
    changed_data = expectation_data()
    changed_data[field] = value
    changed = ReproductionExpectation.model_validate(changed_data)

    fingerprint = compute_reproduction_expectation_fingerprint(original)

    assert len(fingerprint) == 64
    assert fingerprint == fingerprint.lower()
    assert compute_reproduction_expectation_fingerprint(changed) != fingerprint


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_exit_codes", []),
        ("expected_exit_codes", [0]),
        ("expected_exit_codes", [True]),
        ("expected_exit_codes", [1, 1]),
        ("required_fragments", []),
    ],
)
def test_rejects_invalid_expectation_sequences(field: str, value: object) -> None:
    data = expectation_data()
    data[field] = value

    with pytest.raises(ValidationError):
        ReproductionExpectation.model_validate(data)


def test_rejects_duplicate_fragment_ids_across_groups() -> None:
    data = expectation_data()
    data["forbidden_fragments"] = [
        {"fragment_id": "target-failure", "stream": "stderr", "text": "other"}
    ]

    with pytest.raises(ValidationError):
        ReproductionExpectation.model_validate(data)


def test_reproduction_models_reject_unknown_fields() -> None:
    data = expectation_data()
    data["repair_hint"] = "change the implementation"

    with pytest.raises(ValidationError):
        ReproductionExpectation.model_validate(data)


def test_bundle_requires_an_exact_approved_command_id() -> None:
    task = AgentTaskSpec.model_validate(task_data())
    data = expectation_data()
    data["command_id"] = "other_tests"

    with pytest.raises(ValidationError):
        ReproductionTaskBundle(
            task=task,
            reproduction=ReproductionExpectation.model_validate(data),
        )


def test_bundle_agent_view_excludes_reproduction_data() -> None:
    task = AgentTaskSpec.model_validate(task_data())
    bundle = ReproductionTaskBundle(
        task=task,
        reproduction=ReproductionExpectation.model_validate(expectation_data()),
    )

    agent_view = bundle.agent_view()
    rendered = repr(agent_view.model_dump())

    assert agent_view is task
    assert set(agent_view.model_dump()) == set(AgentTaskSpec.model_fields)
    assert "reproduction" not in rendered
    assert "target-failure" not in rendered


def test_evidence_constructor_is_strict_frozen_and_deterministic() -> None:
    evidence = ReproductionEvidence.from_execution_result(execution_result())

    assert evidence.command_id == "unit_tests"
    assert evidence.argv == ("pytest", "-q")
    assert evidence.model_dump() == ReproductionEvidence.from_execution_result(
        execution_result()
    ).model_dump()
    with pytest.raises(ValidationError):
        evidence.stdout = "changed"
    with pytest.raises(ValidationError):
        ReproductionEvidence.model_validate({**evidence.model_dump(), "extra": True})


def test_verdict_enforces_status_invariants() -> None:
    common = {
        "command_id": "unit_tests",
        "reasons": ("fixed evaluator explanation",),
        "matched_required_fragment_ids": ("target",),
        "missing_required_fragment_ids": (),
        "forbidden_fragment_ids_found": (),
    }

    with pytest.raises(ValidationError):
        ReproductionVerdict(
            status=ReproductionStatus.REPRODUCED,
            exit_code=0,
            **common,
        )
    with pytest.raises(ValidationError):
        ReproductionVerdict(
            status=ReproductionStatus.NOT_REPRODUCED,
            exit_code=1,
            **common,
        )
    with pytest.raises(ValidationError):
        ReproductionVerdict(
            status=ReproductionStatus.REPRODUCED,
            exit_code=1,
            **{**common, "missing_required_fragment_ids": ("missing",)},
        )
