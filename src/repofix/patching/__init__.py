"""Controlled, unapplied patch proposal APIs."""

from repofix.patching.context import (
    PatchContextFileObservation,
    PatchProposalContext,
    PatchProposalContextError,
    build_patch_proposal_context,
)
from repofix.patching.models import (
    PATCH_VALIDATION_SUMMARY,
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
    "PatchContextFileObservation",
    "PATCH_VALIDATION_SUMMARY",
    "PatchEditDraft",
    "PatchProposalContext",
    "PatchProposalContextError",
    "PatchProposalDraft",
    "PatchProposalValidationError",
    "PatchProposalValidationStatus",
    "ValidatedPatchEdit",
    "ValidatedPatchFileSnapshot",
    "ValidatedPatchProposal",
    "build_patch_proposal_context",
    "compute_proposal_digest",
    "validate_patch_proposal",
    "validate_patch_workspace_reads",
]
