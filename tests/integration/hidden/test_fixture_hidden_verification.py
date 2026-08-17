"""End-to-end hidden verification for the checked-in evaluator-only fixture."""

from pathlib import Path
import shutil

import pytest

from repofix.agent import ToolExecutionError
from repofix.execution import (
    ApprovedCommandExecutionResult,
    CommandTerminationReason,
    LocalApprovedCommandExecutor,
)
from repofix.hidden import (
    HiddenVerificationStatus,
    resolve_hidden_command,
    verify_hidden_behavior,
)
from repofix.runners import (
    run_hidden_verification_from_paths,
    run_regression_verification_from_paths,
)
from repofix.tasks import load_evaluator_task_bundle
from repofix.tools import LocalReadOnlyToolGateway
from tests.integration.regression.test_fixture_regression_verification import (
    Chain,
    _chain,
)


def _public_regression(chain: Chain):
    return run_regression_verification_from_paths(
        task_path=chain.task_path,
        workspace_root=chain.workspace,
        original_reproduction_result=chain.original,  # type: ignore[arg-type]
        proposal=chain.proposal,  # type: ignore[arg-type]
        baseline_result=chain.baseline,  # type: ignore[arg-type]
        application_result=chain.application,  # type: ignore[arg-type]
        post_patch_reproduction_result=chain.post_patch,  # type: ignore[arg-type]
    )


def _hidden(chain: Chain, regression: object):
    return run_hidden_verification_from_paths(
        task_path=chain.task_path,
        workspace_root=chain.workspace,
        original_reproduction_result=chain.original,  # type: ignore[arg-type]
        proposal=chain.proposal,  # type: ignore[arg-type]
        baseline_result=chain.baseline,  # type: ignore[arg-type]
        application_result=chain.application,  # type: ignore[arg-type]
        post_patch_reproduction_result=chain.post_patch,  # type: ignore[arg-type]
        regression_verification_result=regression,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        (
            "        return configured_value\n",
            HiddenVerificationStatus.HIDDEN_COMMAND_PASSED,
        ),
        (
            '        return "configured"\n',
            HiddenVerificationStatus.HIDDEN_COMMAND_FAILED,
        ),
    ],
    ids=["correct-candidate", "plausible-incomplete-candidate"],
)
def test_publicly_successful_candidates_are_distinguished_by_external_hidden_behavior(
    tmp_path: Path,
    replacement: str,
    expected: HiddenVerificationStatus,
) -> None:
    chain = _chain(tmp_path, start=9, end=9, replacement=replacement)
    regression = _public_regression(chain)
    source_before = (chain.workspace / "src/header_parser.py").read_bytes()
    model_calls = (chain.reproduction_model.calls, chain.patch_model.calls)

    result = _hidden(chain, regression)

    assert result.status is expected
    assert regression.status.value == "regression_command_passed"
    assert result.evidence.argv[-1] == str(
        (
            chain.task_path.parent
            / "hidden_tests/test_header_parser_hidden.py"
        ).resolve()
    )
    assert Path(result.evidence.argv[-1]).is_relative_to(chain.task_path.parent.resolve())
    assert not Path(result.evidence.argv[-1]).is_relative_to(chain.workspace.resolve())
    assert (chain.workspace / "src/header_parser.py").read_bytes() == source_before
    assert (chain.reproduction_model.calls, chain.patch_model.calls) == model_calls == (4, 1)
    if expected is HiddenVerificationStatus.HIDDEN_COMMAND_PASSED:
        assert result.evidence.exit_code == 0
        assert "1 passed" in result.evidence.stdout
    else:
        assert result.evidence.exit_code == 1
        assert "alternate-configured" in (
            result.evidence.stdout + result.evidence.stderr
        )


@pytest.mark.parametrize(
    "termination",
    [CommandTerminationReason.TIMED_OUT, CommandTerminationReason.OUTPUT_LIMIT],
)
def test_valid_public_chain_maps_bounded_hidden_termination_to_inconclusive_once(
    tmp_path: Path,
    termination: CommandTerminationReason,
) -> None:
    chain = _chain(
        tmp_path,
        start=9,
        end=9,
        replacement="        return configured_value\n",
    )
    regression = _public_regression(chain)
    bundle = load_evaluator_task_bundle(chain.task_path)
    resolution = resolve_hidden_command(
        workspace_root=chain.workspace,
        evaluator_assets_root=chain.task_path.parent,
        specification=bundle.hidden_verification,
    )

    class Gateway:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def execute(self, command_id: str) -> ApprovedCommandExecutionResult:
            self.calls.append(command_id)
            return ApprovedCommandExecutionResult(
                command_id=command_id,
                argv=resolution.command.argv,
                termination_reason=termination,
                exit_code=None,
                stdout="bounded evidence\n",
                stderr="",
                stdout_bytes=17,
                stderr_bytes=0,
                had_decode_errors=False,
            )

    gateway = Gateway()
    result = verify_hidden_behavior(
        workspace_root=chain.workspace,
        task=bundle.agent_view(),
        reproduction_expectation=bundle.reproduction,
        regression_specification=bundle.regression,
        hidden_specification=bundle.hidden_verification,
        original_reproduction_result=chain.original,  # type: ignore[arg-type]
        proposal=chain.proposal,  # type: ignore[arg-type]
        regression_baseline_result=chain.baseline,  # type: ignore[arg-type]
        application_result=chain.application,  # type: ignore[arg-type]
        post_patch_reproduction_result=chain.post_patch,  # type: ignore[arg-type]
        regression_verification_result=regression,
        resolved_hidden_command=resolution,
        evaluator_command_gateway=gateway,
    )

    assert result.status is HiddenVerificationStatus.INCONCLUSIVE
    assert gateway.calls == [bundle.hidden_verification.command_id]


def test_checked_in_hidden_fixture_fails_prefixed_behavior_and_passes_trusted_fix(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    task_path = root / "examples/evaluator/empty-header-bug/task.yaml"
    bundle = load_evaluator_task_bundle(task_path)
    outcomes: list[int | None] = []

    for name, corrected in (("pre-fix", False), ("trusted-correct", True)):
        workspace = tmp_path / name
        shutil.copytree(root / "examples/fixtures/empty-header-bug", workspace)
        if corrected:
            source = workspace / "src/header_parser.py"
            source.write_text(
                source.read_text(encoding="utf-8").replace(
                    "return DEFAULT_VALUE",
                    "return configured_value",
                ),
                encoding="utf-8",
            )
        resolution = resolve_hidden_command(
            workspace_root=workspace,
            evaluator_assets_root=task_path.parent,
            specification=bundle.hidden_verification,
        )
        gateway = LocalApprovedCommandExecutor(
            workspace_root=workspace,
            approved_commands={resolution.command_id: resolution.command},
            timeout_seconds=bundle.task.timeout_seconds,
        )
        outcomes.append(gateway.execute(resolution.command_id).exit_code)

    assert outcomes == [1, 0]


def test_agent_repository_tools_cannot_discover_external_hidden_asset(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    task_path = root / "examples/evaluator/empty-header-bug/task.yaml"
    bundle = load_evaluator_task_bundle(task_path)
    workspace = tmp_path / "workspace"
    shutil.copytree(root / "examples/fixtures/empty-header-bug", workspace)
    hidden = task_path.parent / bundle.hidden_verification.test_file.path
    tools = LocalReadOnlyToolGateway(
        workspace_root=workspace,
        allowed_source_paths=bundle.agent_view().allowed_source_paths,
    )

    assert tools.search_code("alternate-configured") == ""
    assert "hidden_tests" not in tools.list_files("tests")
    with pytest.raises(ToolExecutionError):
        tools.read_file(str(hidden.resolve()), 1, 20)
