"""Reusable complete artifact chain for hidden-verification unit tests."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from repofix.hidden import (
    HiddenVerificationSpecification,
    ResolvedHiddenCommand,
    resolve_hidden_command,
)
from repofix.regression import RegressionVerificationResult, verify_post_patch_regression
from repofix.tasks import ApprovedCommand, EvaluatorFileReference
from ..regression.conftest import (
    Gateway,
    PreparedVerification,
    execution,
    prepared_baseline,
    prepared_verification,
)


_IMPORTED_FIXTURES = (prepared_baseline, prepared_verification)


@dataclass
class PreparedHidden:
    public: PreparedVerification
    regression: RegressionVerificationResult
    specification: HiddenVerificationSpecification
    resolution: ResolvedHiddenCommand
    hidden_file: Path


@pytest.fixture
def prepared_hidden(
    prepared_verification: PreparedVerification,
    tmp_path: Path,
) -> PreparedHidden:
    public = prepared_verification
    regression_argv = public.task.approved_commands[public.specification.command_id].argv
    regression = verify_post_patch_regression(
        workspace_root=public.workspace,
        task=public.task,
        reproduction_expectation=public.expectation,
        regression_specification=public.specification,
        original_reproduction_result=public.original,
        proposal=public.proposal,
        baseline_result=public.baseline,
        application_result=public.application,
        post_patch_reproduction_result=public.post_patch,
        command_gateway=Gateway(
            execution(public.specification.command_id, regression_argv)
        ),
    )
    assets_root = tmp_path.parent / f"{tmp_path.name}-evaluator"
    hidden_file = assets_root / "hidden_tests/test_hidden.py"
    hidden_file.parent.mkdir(parents=True)
    contents = b"def test_hidden():\n    assert True\n"
    hidden_file.write_bytes(contents)
    specification = HiddenVerificationSpecification(
        command_id="hidden_tests",
        launcher=ApprovedCommand(argv=("pytest", "-q", "--rootdir", ".")),
        test_file=EvaluatorFileReference(
            path="hidden_tests/test_hidden.py",
            sha256=hashlib.sha256(contents).hexdigest(),
            size_bytes=len(contents),
        ),
    )
    resolution = resolve_hidden_command(
        workspace_root=public.workspace,
        evaluator_assets_root=assets_root,
        specification=specification,
    )
    return PreparedHidden(
        public=public,
        regression=regression,
        specification=specification,
        resolution=resolution,
        hidden_file=hidden_file,
    )
