"""Tests for hidden specifications, fingerprints, and evaluator asset isolation."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from repofix.agent.reproduction_loop import compute_task_fingerprint
from repofix.hidden import (
    HiddenVerificationError,
    HiddenVerificationSpecification,
    compute_hidden_specification_fingerprint,
    resolve_hidden_command,
)
from repofix.tasks import (
    AgentTaskSpec,
    ApprovedCommand,
    EvaluatorFileReference,
    EvaluatorTaskBundle,
    GoldPatchSpec,
    PatchPolicySpecification,
    RegressionSpecification,
)
from repofix.reproduction import ReproductionExpectation


def specification(contents: bytes = b"hidden\n") -> HiddenVerificationSpecification:
    return HiddenVerificationSpecification(
        command_id="hidden_tests",
        launcher=ApprovedCommand(argv=("pytest", "-q", "--rootdir", ".")),
        test_file=EvaluatorFileReference(
            path="hidden_tests/test_hidden.py",
            sha256=hashlib.sha256(contents).hexdigest(),
            size_bytes=len(contents),
        ),
    )


def test_hidden_specification_is_strict_frozen_and_fingerprint_is_field_sensitive() -> None:
    item = specification()
    same = specification()
    changed = item.model_copy(
        update={"launcher": ApprovedCommand(argv=("pytest", "-qq"))}
    )

    assert compute_hidden_specification_fingerprint(item) == (
        compute_hidden_specification_fingerprint(same)
    )
    assert compute_hidden_specification_fingerprint(item) != (
        compute_hidden_specification_fingerprint(changed)
    )
    with pytest.raises(ValidationError):
        item.command_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        HiddenVerificationSpecification.model_validate(
            {**item.model_dump(), "extra": "forbidden"}
        )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/hidden/test.py",
        "../hidden/test.py",
        "hidden/../test.py",
        r"hidden\test.py",
        "hidden/\0test.py",
        "hidden/evil\u2028test.py",
    ],
)
def test_hidden_asset_reference_rejects_ambiguous_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        EvaluatorFileReference(path=path, sha256="a" * 64, size_bytes=1)


@pytest.mark.parametrize(
    ("sha256", "size"),
    [("A" * 64, 1), ("a" * 63, 1), ("g" * 64, 1), ("a" * 64, -1)],
)
def test_hidden_asset_reference_rejects_invalid_identity(
    sha256: str,
    size: int,
) -> None:
    with pytest.raises(ValidationError):
        EvaluatorFileReference(
            path="hidden/test.py",
            sha256=sha256,
            size_bytes=size,
        )


def test_hidden_command_collision_is_rejected_without_affecting_task_fingerprint() -> None:
    task = AgentTaskSpec.model_validate(
        {
            "task_id": "task",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0" * 40,
            "issue_title": "Issue",
            "issue_body": "Body",
            "approved_commands": {"public": {"argv": ["pytest", "-q"]}},
            "allowed_source_paths": ["src"],
            "timeout_seconds": 30,
        }
    )
    before = compute_task_fingerprint(task)
    with pytest.raises(ValidationError, match="agent-visible"):
        EvaluatorTaskBundle(
            task=task,
            reproduction=ReproductionExpectation.model_validate(
                {
                    "command_id": "public",
                    "expected_exit_codes": [1],
                    "required_fragments": [
                        {
                            "fragment_id": "failure",
                            "stream": "combined",
                            "text": "failure",
                        }
                    ],
                }
            ),
            regression=RegressionSpecification(command_id="public"),
            hidden_verification=specification().model_copy(
                update={"command_id": "public"}
            ),
            patch_policy=PatchPolicySpecification(protected_files=()),
            gold_patch=GoldPatchSpec(patch="diff"),
        )
    assert compute_task_fingerprint(task) == before


def _layout(tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    workspace = tmp_path / "workspace"
    assets = tmp_path / "evaluator"
    hidden = assets / "hidden_tests/test_hidden.py"
    workspace.mkdir()
    hidden.parent.mkdir(parents=True)
    contents = b"hidden\n"
    hidden.write_bytes(contents)
    return workspace, assets, hidden, contents


def test_resolved_hidden_command_uses_external_absolute_asset_and_exact_launcher(
    tmp_path: Path,
) -> None:
    workspace, assets, hidden, contents = _layout(tmp_path)

    result = resolve_hidden_command(
        workspace_root=workspace,
        evaluator_assets_root=assets,
        specification=specification(contents),
    )

    assert result.workspace == workspace.resolve()
    assert result.evaluator_assets_root == assets.resolve()
    assert result.hidden_file.resolved == hidden.resolve()
    assert result.hidden_file.resolved.is_relative_to(assets.resolve())
    assert not result.hidden_file.resolved.is_relative_to(workspace.resolve())
    assert result.command.argv == (
        "pytest",
        "-q",
        "--rootdir",
        ".",
        str(hidden.resolve()),
    )


def test_evaluator_root_inside_workspace_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    assets = workspace / "evaluator"
    hidden = assets / "hidden_tests/test_hidden.py"
    hidden.parent.mkdir(parents=True)
    contents = b"hidden\n"
    hidden.write_bytes(contents)

    with pytest.raises(HiddenVerificationError, match="outside"):
        resolve_hidden_command(
            workspace_root=workspace,
            evaluator_assets_root=assets,
            specification=specification(contents),
        )


@pytest.mark.parametrize("mismatch", ["hash", "size"])
def test_hidden_file_identity_mismatch_is_rejected(
    tmp_path: Path,
    mismatch: str,
) -> None:
    workspace, assets, _, contents = _layout(tmp_path)
    item = specification(contents)
    if mismatch == "hash":
        reference = item.test_file.model_copy(update={"sha256": "f" * 64})
    else:
        reference = item.test_file.model_copy(
            update={"size_bytes": item.test_file.size_bytes + 1}
        )
    item = item.model_copy(update={"test_file": reference})

    with pytest.raises(HiddenVerificationError, match="does not match"):
        resolve_hidden_command(
            workspace_root=workspace,
            evaluator_assets_root=assets,
            specification=item,
        )


def test_hidden_symlink_escape_and_resolution_into_workspace_are_rejected(
    tmp_path: Path,
) -> None:
    workspace, assets, hidden, contents = _layout(tmp_path)
    workspace_target = workspace / "hidden.py"
    workspace_target.write_bytes(contents)
    hidden.unlink()
    try:
        hidden.symlink_to(workspace_target)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symbolic links are not supported on this host: {error}")

    with pytest.raises(HiddenVerificationError, match="safely inspected"):
        resolve_hidden_command(
            workspace_root=workspace,
            evaluator_assets_root=assets,
            specification=specification(contents),
        )
