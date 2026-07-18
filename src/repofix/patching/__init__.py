"""Controlled, unapplied patch proposal APIs."""

from repofix.patching.application import (
    PatchApplicationError,
    apply_validated_patch_proposal,
)
from repofix.patching.context import (
    PatchContextFileObservation,
    PatchProposalContext,
    PatchProposalContextError,
    build_patch_proposal_context,
)
from repofix.patching.models import (
    PATCH_APPLICATION_SUMMARY,
    PATCH_VALIDATION_SUMMARY,
    AppliedPatchFile,
    PatchApplicationResult,
    PatchApplicationStatus,
    PatchEditDraft,
    PatchProposalDraft,
    PatchProposalValidationStatus,
    ValidatedPatchEdit,
    ValidatedPatchFileSnapshot,
    ValidatedPatchProposal,
    compute_proposal_digest,
)
from repofix.patching.validator import (
    PatchProposalValidationError,
    validate_patch_proposal,
    validate_patch_workspace_reads,
)

__all__ = [
    "AppliedPatchFile",
    "PATCH_APPLICATION_SUMMARY",
    "PatchContextFileObservation",
    "PATCH_VALIDATION_SUMMARY",
    "PatchEditDraft",
    "PatchApplicationError",
    "PatchApplicationResult",
    "PatchApplicationStatus",
    "PatchProposalContext",
    "PatchProposalContextError",
    "PatchProposalDraft",
    "PatchProposalValidationError",
    "PatchProposalValidationStatus",
    "ValidatedPatchEdit",
    "ValidatedPatchFileSnapshot",
    "ValidatedPatchProposal",
    "apply_validated_patch_proposal",
    "build_patch_proposal_context",
    "compute_proposal_digest",
    "validate_patch_proposal",
    "validate_patch_workspace_reads",
]
