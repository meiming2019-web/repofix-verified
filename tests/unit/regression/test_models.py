"""Tests for strict regression specifications and system-owned results."""

import pytest
from pydantic import ValidationError

from repofix.regression import (
    RegressionSpecification,
    compute_regression_baseline_fingerprint,
    compute_regression_specification_fingerprint,
    compute_regression_verification_fingerprint,
    verify_post_patch_regression,
)
from repofix.patching import compute_patch_application_result_fingerprint
from repofix.reproduction import compute_post_patch_reproduction_fingerprint

from .conftest import Gateway, PreparedVerification, execution


def test_regression_specification_is_strict_frozen_and_fingerprinted() -> None:
    specification = RegressionSpecification(command_id="regression_tests")
    fingerprint = compute_regression_specification_fingerprint(specification)

    assert len(fingerprint) == 64
    assert fingerprint == compute_regression_specification_fingerprint(specification)
    assert fingerprint != compute_regression_specification_fingerprint(
        RegressionSpecification(command_id="other_tests")
    )
    with pytest.raises(ValidationError):
        specification.command_id = "other_tests"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        RegressionSpecification.model_validate(
            {"command_id": "regression_tests", "extra": True}
        )
    with pytest.raises(ValidationError):
        RegressionSpecification(command_id="")


def test_baseline_and_verification_results_are_frozen_and_fingerprinted(
    prepared_verification: PreparedVerification,
) -> None:
    baseline = prepared_verification.baseline
    fingerprint = compute_regression_baseline_fingerprint(baseline)
    gateway = Gateway(
        execution(
            "regression_tests",
            prepared_verification.task.approved_commands["regression_tests"].argv,
        )
    )
    verification = verify_post_patch_regression(
        workspace_root=prepared_verification.workspace,
        task=prepared_verification.task,
        reproduction_expectation=prepared_verification.expectation,
        regression_specification=prepared_verification.specification,
        original_reproduction_result=prepared_verification.original,
        proposal=prepared_verification.proposal,
        baseline_result=baseline,
        application_result=prepared_verification.application,
        post_patch_reproduction_result=prepared_verification.post_patch,
        command_gateway=gateway,
    )

    assert len(fingerprint) == 64
    assert fingerprint == compute_regression_baseline_fingerprint(baseline)
    assert fingerprint != compute_regression_baseline_fingerprint(
        baseline.model_copy(update={"task_id": "changed"})
    )
    application_fingerprint = compute_patch_application_result_fingerprint(
        prepared_verification.application
    )
    assert application_fingerprint != compute_patch_application_result_fingerprint(
        prepared_verification.application.model_copy(update={"task_id": "changed"})
    )
    post_patch_fingerprint = compute_post_patch_reproduction_fingerprint(
        prepared_verification.post_patch
    )
    assert post_patch_fingerprint != compute_post_patch_reproduction_fingerprint(
        prepared_verification.post_patch.model_copy(update={"task_id": "changed"})
    )
    verification_fingerprint = compute_regression_verification_fingerprint(
        verification
    )
    assert verification_fingerprint == compute_regression_verification_fingerprint(
        verification
    )
    assert verification_fingerprint != compute_regression_verification_fingerprint(
        verification.model_copy(update={"task_id": "changed"})
    )
    with pytest.raises(ValidationError):
        baseline.status = baseline.status  # type: ignore[misc]
    with pytest.raises(ValidationError):
        verification.status = verification.status  # type: ignore[misc]
    with pytest.raises(ValidationError):
        type(verification).model_validate({**verification.model_dump(), "extra": True})
