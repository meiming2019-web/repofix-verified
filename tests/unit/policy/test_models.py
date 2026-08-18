"""Tests for strict policy specifications, findings, and fingerprints."""

import pytest
from pydantic import ValidationError

from repofix.hidden import compute_hidden_verification_fingerprint
from repofix.policy import (
    FORBIDDEN_PATH_PRESENT_SUMMARY,
    PatchPolicySpecification,
    PolicyFinding,
    PolicyRuleId,
    WorkspaceFileReference,
    compute_patch_policy_specification_fingerprint,
    verify_patch_policy,
)
from .conftest import PreparedPolicy


def reference(path: str = "tests/protected.py", **updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "path": path,
        "sha256": "a" * 64,
        "size_bytes": 10,
    }
    values.update(updates)
    return values


def test_policy_specification_is_strict_frozen_canonical_and_fingerprintable() -> None:
    specification = PatchPolicySpecification.model_validate(
        {
            "protected_files": [reference("z.py"), reference("a.py")],
            "forbidden_paths": ["z-forbidden.py", "a-forbidden.py"],
        }
    )

    assert tuple(item.path for item in specification.protected_files) == ("a.py", "z.py")
    assert specification.forbidden_paths == ("a-forbidden.py", "z-forbidden.py")
    assert compute_patch_policy_specification_fingerprint(specification) == (
        compute_patch_policy_specification_fingerprint(specification)
    )
    with pytest.raises(ValidationError):
        PatchPolicySpecification.model_validate(
            {"protected_files": [], "unknown": True}
        )
    with pytest.raises(ValidationError):
        specification.forbidden_paths = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "path",
    ["", "/tests/a.py", "tests/../a.py", r"tests\a.py", "tests//a.py", "tests/a.py\0"],
)
def test_policy_paths_reject_unsafe_or_ambiguous_forms(path: str) -> None:
    with pytest.raises(ValidationError):
        WorkspaceFileReference.model_validate(reference(path))
    with pytest.raises(ValidationError):
        PatchPolicySpecification.model_validate(
            {"protected_files": [], "forbidden_paths": [path]}
        )


@pytest.mark.parametrize(
    "values",
    [
        {"protected_files": [reference(), reference()]},
        {"protected_files": [], "forbidden_paths": ["a.py", "a.py"]},
        {"protected_files": [reference("a.py")], "forbidden_paths": ["a.py"]},
    ],
)
def test_policy_paths_must_be_unique_and_disjoint(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PatchPolicySpecification.model_validate(values)


@pytest.mark.parametrize(
    "updates",
    [
        {"sha256": "A" * 64},
        {"sha256": "a" * 63},
        {"size_bytes": -1},
        {"size_bytes": True},
    ],
)
def test_protected_file_identity_is_strict(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorkspaceFileReference.model_validate(reference(**updates))


def test_policy_fingerprint_is_field_sensitive() -> None:
    first = PatchPolicySpecification.model_validate(
        {"protected_files": [reference()], "forbidden_paths": ["conftest.py"]}
    )
    second = first.model_copy(update={"forbidden_paths": ("tests/conftest.py",)})

    assert compute_patch_policy_specification_fingerprint(first) != (
        compute_patch_policy_specification_fingerprint(second)
    )


def test_hidden_verification_fingerprint_is_deterministic_and_field_sensitive(
    prepared_policy: PreparedPolicy,
) -> None:
    result = prepared_policy.hidden_result

    assert compute_hidden_verification_fingerprint(result) == (
        compute_hidden_verification_fingerprint(result)
    )
    changed = result.model_copy(update={"verification_summary": "changed"})
    assert compute_hidden_verification_fingerprint(result) != (
        compute_hidden_verification_fingerprint(changed)
    )


def test_findings_require_safe_paths_and_fixed_system_summary() -> None:
    finding = PolicyFinding(
        rule_id=PolicyRuleId.FORBIDDEN_PATH_PRESENT,
        path="tests/conftest.py",
        summary=FORBIDDEN_PATH_PRESENT_SUMMARY,
    )
    assert finding.summary == FORBIDDEN_PATH_PRESENT_SUMMARY
    with pytest.raises(ValidationError):
        finding.summary = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        PolicyFinding(
            rule_id=PolicyRuleId.FORBIDDEN_PATH_PRESENT,
            path="tests/conftest.py",
            summary="model text",
        )


def test_policy_result_is_immutable_and_status_matches_findings(
    prepared_policy: PreparedPolicy,
) -> None:
    chain = prepared_policy.hidden_chain
    public = chain.public
    result = verify_patch_policy(
        workspace_root=public.workspace,
        task=public.task,
        policy_specification=prepared_policy.specification,
        proposal=public.proposal,
        application_result=public.application,
        post_patch_reproduction_result=public.post_patch,
        regression_verification_result=chain.regression,
        hidden_verification_result=prepared_policy.hidden_result,
    )
    with pytest.raises(ValidationError):
        result.status = result.status  # type: ignore[misc]
    with pytest.raises(ValidationError):
        type(result).model_validate(
            {**result.model_dump(), "status": "policy_failed"}
        )
