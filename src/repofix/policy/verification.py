"""Pure protected-workspace policy verification."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Never, TypeVar

from pydantic import BaseModel, ValidationError

from repofix.agent.reproduction_loop import compute_task_fingerprint
from repofix.hidden.models import (
    HiddenVerificationResult,
    HiddenVerificationStatus,
    compute_hidden_verification_fingerprint,
)
from repofix.patching.models import (
    PatchApplicationResult,
    PatchApplicationStatus,
    ValidatedPatchProposal,
    compute_patch_application_result_fingerprint,
)
from repofix.policy.errors import PolicyVerificationError
from repofix.policy.models import (
    PolicyFinding,
    PolicyRuleId,
    PolicyVerificationResult,
    PolicyVerificationStatus,
    compute_patch_policy_specification_fingerprint,
    finding_summary_for,
    policy_summary_for,
)
from repofix.regression._workspace import capture_applied_targets, resolve_workspace
from repofix.regression.models import (
    RegressionVerificationResult,
    RegressionVerificationStatus,
    compute_regression_verification_fingerprint,
)
from repofix.reproduction.post_patch import (
    PostPatchReproductionResult,
    PostPatchReproductionStatus,
    compute_post_patch_reproduction_fingerprint,
)
from repofix.tasks.spec import (
    AgentTaskSpec,
    PatchPolicySpecification,
    WorkspaceFileReference,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _fail(message: str, cause: BaseException | None = None) -> Never:
    if cause is None:
        raise PolicyVerificationError(message)
    raise PolicyVerificationError(message) from cause


def _canonical(value: object, model: type[ModelT], description: str) -> ModelT:
    if not isinstance(value, model):
        _fail(f"{description} has an invalid type")
    try:
        return model.model_validate(value.model_dump())
    except ValidationError as error:
        _fail(f"{description} failed canonical integrity checks", error)


def _verify_chain(
    *,
    task: AgentTaskSpec,
    policy: PatchPolicySpecification,
    proposal: ValidatedPatchProposal,
    application: PatchApplicationResult,
    post_patch: PostPatchReproductionResult,
    regression: RegressionVerificationResult,
    hidden: HiddenVerificationResult,
) -> tuple[str, str, str, str, str, str]:
    task_fingerprint = compute_task_fingerprint(task)
    policy_fingerprint = compute_patch_policy_specification_fingerprint(policy)
    application_fingerprint = compute_patch_application_result_fingerprint(application)
    post_patch_fingerprint = compute_post_patch_reproduction_fingerprint(post_patch)
    regression_fingerprint = compute_regression_verification_fingerprint(regression)
    hidden_fingerprint = compute_hidden_verification_fingerprint(hidden)

    if any(
        value != task.task_id
        for value in (
            proposal.task_id,
            application.task_id,
            post_patch.task_id,
            regression.task_id,
            hidden.task_id,
        )
    ):
        _fail("policy verification task identity does not match")
    if any(
        value != task_fingerprint
        for value in (
            proposal.task_fingerprint,
            application.task_fingerprint,
            post_patch.task_fingerprint,
            regression.task_fingerprint,
            hidden.task_fingerprint,
        )
    ):
        _fail("policy verification task fingerprint does not match")
    if any(
        value != proposal.proposal_digest
        for value in (
            application.proposal_digest,
            post_patch.proposal_digest,
            regression.proposal_digest,
            hidden.proposal_digest,
        )
    ):
        _fail("policy verification proposal digest does not match")
    if any(
        value != proposal.reproduction_expectation_fingerprint
        for value in (
            application.reproduction_expectation_fingerprint,
            post_patch.reproduction_expectation_fingerprint,
            regression.reproduction_expectation_fingerprint,
            hidden.reproduction_expectation_fingerprint,
        )
    ):
        _fail("policy verification reproduction expectation binding does not match")
    if any(
        value != proposal.reproduction_run_fingerprint
        for value in (
            application.reproduction_run_fingerprint,
            post_patch.original_reproduction_run_fingerprint,
            regression.original_reproduction_run_fingerprint,
            hidden.original_reproduction_run_fingerprint,
        )
    ):
        _fail("policy verification reproduction run binding does not match")
    if (
        regression.application_result_fingerprint != application_fingerprint
        or hidden.application_result_fingerprint != application_fingerprint
    ):
        _fail("policy verification application fingerprint does not match")
    if (
        regression.post_patch_reproduction_fingerprint != post_patch_fingerprint
        or hidden.post_patch_reproduction_fingerprint != post_patch_fingerprint
    ):
        _fail("policy verification post-patch fingerprint does not match")
    if hidden.regression_verification_fingerprint != regression_fingerprint:
        _fail("policy verification regression fingerprint does not match")
    if hidden.regression_baseline_fingerprint != regression.regression_baseline_fingerprint:
        _fail("policy verification regression baseline binding does not match")
    if application.status is not PatchApplicationStatus.APPLIED:
        _fail("policy verification requires an applied patch result")
    if any(
        value is not application.status
        for value in (
            post_patch.application_status,
            regression.application_status,
            hidden.application_status,
        )
    ):
        _fail("policy verification application statuses do not match")
    if (
        post_patch.status
        is not PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED
        or regression.post_patch_reproduction_status is not post_patch.status
        or hidden.post_patch_reproduction_status is not post_patch.status
    ):
        _fail("policy verification requires the post-patch reproduction gate")
    if (
        regression.status
        is not RegressionVerificationStatus.REGRESSION_COMMAND_PASSED
        or hidden.regression_verification_status is not regression.status
    ):
        _fail("policy verification requires the public regression pass gate")
    if hidden.status is not HiddenVerificationStatus.HIDDEN_COMMAND_PASSED:
        _fail("policy verification requires the hidden pass gate")
    return (
        task_fingerprint,
        policy_fingerprint,
        application_fingerprint,
        post_patch_fingerprint,
        regression_fingerprint,
        hidden_fingerprint,
    )


def _parents_are_safe(workspace: Path, logical: PurePosixPath) -> bool:
    current = workspace
    for component in logical.parts[:-1]:
        current /= component
        try:
            metadata = current.lstat()
        except (FileNotFoundError, NotADirectoryError):
            return False
        except OSError as error:
            _fail("policy-controlled path could not be safely inspected", error)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            return False
    return True


def _metadata(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode


def _protected_file_matches(
    *, workspace: Path, reference: WorkspaceFileReference
) -> bool:
    logical = PurePosixPath(reference.path)
    if not _parents_are_safe(workspace, logical):
        return False
    local = workspace.joinpath(*logical.parts)
    try:
        pre_lstat = local.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as error:
        _fail("protected workspace file could not be safely inspected", error)
    if stat.S_ISLNK(pre_lstat.st_mode) or not stat.S_ISREG(pre_lstat.st_mode):
        return False
    if pre_lstat.st_size != reference.size_bytes:
        return False

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    contents = b""
    try:
        try:
            descriptor = os.open(local, flags)
        except FileNotFoundError:
            return False
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                return False
            _fail("protected workspace file could not be safely opened", error)
        pre_fstat = os.fstat(descriptor)
        if not stat.S_ISREG(pre_fstat.st_mode):
            return False
        if _metadata(pre_lstat) != _metadata(pre_fstat):
            _fail("protected workspace file identity changed during inspection")
        remaining = reference.size_bytes + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        contents = b"".join(chunks)
        post_fstat = os.fstat(descriptor)
    except PolicyVerificationError:
        raise
    except OSError as error:
        _fail("protected workspace file could not be safely read", error)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as error:
                _fail("protected workspace file descriptor could not be closed", error)

    if not _parents_are_safe(workspace, logical):
        return False
    try:
        post_lstat = local.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as error:
        _fail("protected workspace file could not be safely re-inspected", error)
    if stat.S_ISLNK(post_lstat.st_mode) or not stat.S_ISREG(post_lstat.st_mode):
        return False
    if (
        _metadata(pre_lstat) != _metadata(post_lstat)
        or _metadata(pre_lstat) != _metadata(post_fstat)
    ):
        _fail("protected workspace file identity changed during inspection")
    return (
        len(contents) == reference.size_bytes
        and hashlib.sha256(contents).hexdigest() == reference.sha256
    )


def _forbidden_path_exists(*, workspace: Path, path: str) -> bool:
    logical = PurePosixPath(path)
    if not _parents_are_safe(workspace, logical):
        return False
    local = workspace.joinpath(*logical.parts)
    try:
        local.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as error:
        _fail("forbidden workspace path could not be safely inspected", error)
    return True


def _inspect_policy(
    *, workspace: Path, specification: PatchPolicySpecification
) -> tuple[PolicyFinding, ...]:
    findings: list[PolicyFinding] = []
    for reference in specification.protected_files:
        if not _protected_file_matches(workspace=workspace, reference=reference):
            rule = PolicyRuleId.PROTECTED_FILE_INTEGRITY
            findings.append(
                PolicyFinding(
                    rule_id=rule,
                    path=reference.path,
                    summary=finding_summary_for(rule),
                )
            )
    for path in specification.forbidden_paths:
        if _forbidden_path_exists(workspace=workspace, path=path):
            rule = PolicyRuleId.FORBIDDEN_PATH_PRESENT
            findings.append(
                PolicyFinding(
                    rule_id=rule,
                    path=path,
                    summary=finding_summary_for(rule),
                )
            )
    return tuple(sorted(findings, key=lambda item: (item.rule_id.value, item.path)))


def verify_patch_policy(
    *,
    workspace_root: Path,
    task: AgentTaskSpec,
    policy_specification: PatchPolicySpecification,
    proposal: ValidatedPatchProposal,
    application_result: PatchApplicationResult,
    post_patch_reproduction_result: PostPatchReproductionResult,
    regression_verification_result: RegressionVerificationResult,
    hidden_verification_result: HiddenVerificationResult,
) -> PolicyVerificationResult:
    """Validate the upstream chain and inspect exact policy-controlled paths."""
    canonical_task = _canonical(task, AgentTaskSpec, "current task")
    canonical_policy = _canonical(
        policy_specification, PatchPolicySpecification, "patch policy specification"
    )
    canonical_proposal = _canonical(
        proposal, ValidatedPatchProposal, "validated proposal"
    )
    canonical_application = _canonical(
        application_result, PatchApplicationResult, "patch application result"
    )
    canonical_post_patch = _canonical(
        post_patch_reproduction_result,
        PostPatchReproductionResult,
        "post-patch reproduction result",
    )
    canonical_regression = _canonical(
        regression_verification_result,
        RegressionVerificationResult,
        "regression verification result",
    )
    canonical_hidden = _canonical(
        hidden_verification_result,
        HiddenVerificationResult,
        "hidden verification result",
    )
    (
        task_fingerprint,
        policy_fingerprint,
        application_fingerprint,
        post_patch_fingerprint,
        regression_fingerprint,
        hidden_fingerprint,
    ) = _verify_chain(
        task=canonical_task,
        policy=canonical_policy,
        proposal=canonical_proposal,
        application=canonical_application,
        post_patch=canonical_post_patch,
        regression=canonical_regression,
        hidden=canonical_hidden,
    )
    workspace = resolve_workspace(
        workspace_root,
        error_type=PolicyVerificationError,
        stage="policy verification",
    )
    capture_applied_targets(
        workspace=workspace,
        proposal=canonical_proposal,
        application=canonical_application,
        error_factory=PolicyVerificationError,
        stage="policy verification",
    )
    findings = _inspect_policy(workspace=workspace, specification=canonical_policy)
    status = (
        PolicyVerificationStatus.POLICY_FAILED
        if findings
        else PolicyVerificationStatus.POLICY_PASSED
    )
    return PolicyVerificationResult(
        task_id=canonical_task.task_id,
        task_fingerprint=task_fingerprint,
        policy_specification_fingerprint=policy_fingerprint,
        proposal_digest=canonical_proposal.proposal_digest,
        application_result_fingerprint=application_fingerprint,
        post_patch_reproduction_fingerprint=post_patch_fingerprint,
        regression_verification_fingerprint=regression_fingerprint,
        hidden_verification_fingerprint=hidden_fingerprint,
        application_status=canonical_application.status,
        post_patch_reproduction_status=canonical_post_patch.status,
        regression_verification_status=canonical_regression.status,
        hidden_verification_status=canonical_hidden.status,
        status=status,
        findings=findings,
        verification_summary=policy_summary_for(status),
    )
