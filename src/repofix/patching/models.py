"""Strict models for bounded, unapplied patch proposals."""

import hashlib
import json
import re
from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator, model_validator

from repofix.tasks.spec import StrictFrozenModel, validate_relative_source_path


MAX_PATCH_FILES = 3
MAX_PATCH_EDITS = 8
MAX_REPLACEMENT_CHARS = 12_000
MAX_TOTAL_REPLACEMENT_CHARS = 24_000
MAX_PATCH_SUMMARY_CHARS = 1_000
MAX_PATCH_RATIONALE_CHARS = 1_000
MAX_PATCH_DIFF_CHARS = 50_000
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PatchEditDraft(StrictFrozenModel):
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    replacement_text: str = Field(max_length=MAX_REPLACEMENT_CHARS)
    rationale: str = Field(max_length=MAX_PATCH_RATIONALE_CHARS)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_relative_source_path(value, description="patch edit path")

    @field_validator("start_line", "end_line", mode="before")
    @classmethod
    def reject_boolean_lines(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("line numbers must not be booleans")
        return value

    @field_validator("replacement_text")
    @classmethod
    def validate_replacement(cls, value: str) -> str:
        if "\0" in value:
            raise ValueError("replacement text must not contain NUL bytes")
        return value

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("edit rationale must not be empty")
        return value

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("edit end line must not precede start line")
        return self


class PatchProposalDraft(StrictFrozenModel):
    hypothesis_id: str
    model_summary: str = Field(min_length=1, max_length=MAX_PATCH_SUMMARY_CHARS)
    edits: tuple[PatchEditDraft, ...] = Field(min_length=1, max_length=MAX_PATCH_EDITS)

    @field_validator("edits", mode="before")
    @classmethod
    def normalize_edits(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("patch edits must be a list or tuple")
        return tuple(value)

    @field_validator("hypothesis_id", "model_summary")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("proposal identifiers and summary must not be empty")
        if "\0" in value:
            raise ValueError("proposal identifiers and summary must not contain NUL bytes")
        return value

    @model_validator(mode="after")
    def validate_edits(self) -> Self:
        if not self.edits:
            raise ValueError("at least one patch edit is required")
        if len(self.edits) > MAX_PATCH_EDITS:
            raise ValueError("patch proposal exceeds the edit limit")
        if len({edit.path for edit in self.edits}) > MAX_PATCH_FILES:
            raise ValueError("patch proposal exceeds the file limit")
        keys = tuple((edit.path, edit.start_line, edit.end_line) for edit in self.edits)
        if len(keys) != len(set(keys)):
            raise ValueError("patch proposal contains duplicate edits")
        if keys != tuple(sorted(keys)):
            raise ValueError("patch edits must use deterministic path and range ordering")
        if sum(len(edit.replacement_text) for edit in self.edits) > MAX_TOTAL_REPLACEMENT_CHARS:
            raise ValueError("patch proposal exceeds the total replacement limit")
        return self


class ValidatedPatchEdit(StrictFrozenModel):
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    original_text: str
    replacement_text: str = Field(max_length=MAX_REPLACEMENT_CHARS)
    original_file_sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_relative_source_path(value, description="validated edit path")

    @field_validator("original_file_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("original file SHA-256 must be lowercase hexadecimal")
        return value

    @field_validator("original_text", "replacement_text")
    @classmethod
    def reject_nul(cls, value: str) -> str:
        if "\0" in value:
            raise ValueError("validated patch text must not contain NUL bytes")
        return value

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("validated edit range is invalid")
        if self.original_text == self.replacement_text:
            raise ValueError("validated patch edits must change their source text")
        return self


class ValidatedPatchFileSnapshot(StrictFrozenModel):
    """System-owned source identity bound to all edits for one logical file."""

    path: str
    original_file_sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return validate_relative_source_path(value, description="validated snapshot path")

    @field_validator("original_file_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("snapshot SHA-256 must be lowercase hexadecimal")
        return value


class PatchProposalValidationStatus(StrEnum):
    STRUCTURALLY_VALIDATED_UNAPPLIED = "structurally_validated_unapplied"


PATCH_VALIDATION_SUMMARY = (
    "Patch proposal passed structural validation. It has not been applied or tested."
)


class ValidatedPatchProposal(StrictFrozenModel):
    task_id: str
    task_fingerprint: str
    reproduction_expectation_fingerprint: str
    hypothesis_id: str
    model_summary: str = Field(min_length=1, max_length=MAX_PATCH_SUMMARY_CHARS)
    validation_status: PatchProposalValidationStatus
    validation_summary: str
    edits: tuple[ValidatedPatchEdit, ...] = Field(min_length=1, max_length=MAX_PATCH_EDITS)
    file_snapshots: tuple[ValidatedPatchFileSnapshot, ...] = Field(
        min_length=1, max_length=MAX_PATCH_FILES
    )
    unified_diff: str = Field(min_length=1, max_length=MAX_PATCH_DIFF_CHARS)
    proposal_digest: str

    @field_validator(
        "proposal_digest",
        "task_fingerprint",
        "reproduction_expectation_fingerprint",
    )
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("proposal hashes must be lowercase hexadecimal SHA-256")
        return value

    @field_validator(
        "task_id",
        "hypothesis_id",
        "model_summary",
        "validation_summary",
        "unified_diff",
    )
    @classmethod
    def reject_nul_text(cls, value: str) -> str:
        if "\0" in value:
            raise ValueError("validated proposal text must not contain NUL bytes")
        if not value.strip():
            raise ValueError("validated proposal text must not be blank")
        return value

    @field_validator("edits", "file_snapshots", mode="before")
    @classmethod
    def normalize_edits(cls, value: object) -> tuple[object, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("validated edits must be a list or tuple")
        return tuple(value)

    @model_validator(mode="after")
    def validate_integrity(self) -> Self:
        if not self.edits:
            raise ValueError("validated proposals require at least one edit")
        keys = tuple((edit.path, edit.start_line, edit.end_line) for edit in self.edits)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("validated edits must be sorted and unique")
        edit_paths = {edit.path for edit in self.edits}
        if len(edit_paths) > MAX_PATCH_FILES:
            raise ValueError("validated proposal exceeds the file limit")
        hashes: dict[str, str] = {}
        for edit in self.edits:
            previous = hashes.setdefault(edit.path, edit.original_file_sha256)
            if previous != edit.original_file_sha256:
                raise ValueError("edits for one file must share the source hash")
        if sum(len(edit.replacement_text) for edit in self.edits) > MAX_TOTAL_REPLACEMENT_CHARS:
            raise ValueError("validated proposal exceeds the total replacement limit")
        snapshot_paths = tuple(snapshot.path for snapshot in self.file_snapshots)
        if snapshot_paths != tuple(sorted(snapshot_paths)) or len(snapshot_paths) != len(
            set(snapshot_paths)
        ):
            raise ValueError("validated file snapshots must be sorted and unique")
        if set(snapshot_paths) != edit_paths:
            raise ValueError("validated proposals require one snapshot per edited path")
        snapshots = {snapshot.path: snapshot for snapshot in self.file_snapshots}
        if any(
            edit.original_file_sha256 != snapshots[edit.path].original_file_sha256
            for edit in self.edits
        ):
            raise ValueError("validated edit hashes must match their file snapshots")
        if not self.unified_diff or "\0" in self.unified_diff:
            raise ValueError("validated proposals require a nonempty NUL-free diff")
        if (
            self.validation_status
            is not PatchProposalValidationStatus.STRUCTURALLY_VALIDATED_UNAPPLIED
        ):
            raise ValueError("validated proposals require the canonical validation status")
        if self.validation_summary != PATCH_VALIDATION_SUMMARY:
            raise ValueError("validated proposals require the canonical validation summary")
        expected = compute_proposal_digest(
            task_id=self.task_id,
            task_fingerprint=self.task_fingerprint,
            reproduction_expectation_fingerprint=(self.reproduction_expectation_fingerprint),
            hypothesis_id=self.hypothesis_id,
            model_summary=self.model_summary,
            validation_status=self.validation_status,
            validation_summary=self.validation_summary,
            edits=self.edits,
            file_snapshots=self.file_snapshots,
            unified_diff=self.unified_diff,
        )
        if self.proposal_digest != expected:
            raise ValueError("proposal digest does not match canonical proposal fields")
        return self


def compute_proposal_digest(
    *,
    task_id: str,
    task_fingerprint: str,
    reproduction_expectation_fingerprint: str,
    hypothesis_id: str,
    model_summary: str,
    validation_status: PatchProposalValidationStatus,
    validation_summary: str,
    edits: tuple[ValidatedPatchEdit, ...],
    file_snapshots: tuple[ValidatedPatchFileSnapshot, ...],
    unified_diff: str,
) -> str:
    """Return deterministic integrity metadata, not an authenticity guarantee."""
    canonical = json.dumps(
        {
            "edits": [edit.model_dump(mode="json") for edit in edits],
            "file_snapshots": [snapshot.model_dump(mode="json") for snapshot in file_snapshots],
            "hypothesis_id": hypothesis_id,
            "model_summary": model_summary,
            "task_id": task_id,
            "task_fingerprint": task_fingerprint,
            "reproduction_expectation_fingerprint": (reproduction_expectation_fingerprint),
            "unified_diff": unified_diff,
            "validation_status": validation_status.value,
            "validation_summary": validation_summary,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
