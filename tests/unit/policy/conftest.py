"""Reusable passing provenance chain for policy verification tests."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from repofix.hidden import HiddenVerificationResult, verify_hidden_behavior
from repofix.policy import PatchPolicySpecification, WorkspaceFileReference
from ..hidden.conftest import PreparedHidden, prepared_hidden
from ..regression.conftest import (
    Gateway,
    execution,
    prepared_baseline,
    prepared_verification,
)


_IMPORTED_FIXTURES = (prepared_baseline, prepared_verification, prepared_hidden)


@dataclass
class PreparedPolicy:
    hidden_chain: PreparedHidden
    hidden_result: HiddenVerificationResult
    specification: PatchPolicySpecification
    protected_file: Path


@pytest.fixture
def prepared_policy(prepared_hidden: PreparedHidden) -> PreparedPolicy:
    public = prepared_hidden.public
    resolution = prepared_hidden.resolution
    hidden_result = verify_hidden_behavior(
        workspace_root=public.workspace,
        task=public.task,
        reproduction_expectation=public.expectation,
        regression_specification=public.specification,
        hidden_specification=prepared_hidden.specification,
        original_reproduction_result=public.original,
        proposal=public.proposal,
        regression_baseline_result=public.baseline,
        application_result=public.application,
        post_patch_reproduction_result=public.post_patch,
        regression_verification_result=prepared_hidden.regression,
        resolved_hidden_command=resolution,
        evaluator_command_gateway=Gateway(
            execution(resolution.command_id, resolution.command.argv)
        ),
    )
    protected_file = public.workspace / "tests/protected.py"
    protected_file.parent.mkdir()
    contents = b"protected fixture\n"
    protected_file.write_bytes(contents)
    specification = PatchPolicySpecification(
        protected_files=(
            WorkspaceFileReference(
                path="tests/protected.py",
                sha256=hashlib.sha256(contents).hexdigest(),
                size_bytes=len(contents),
            ),
        ),
        forbidden_paths=("conftest.py", "tests/conftest.py"),
    )
    return PreparedPolicy(
        hidden_chain=prepared_hidden,
        hidden_result=hidden_result,
        specification=specification,
        protected_file=protected_file,
    )
