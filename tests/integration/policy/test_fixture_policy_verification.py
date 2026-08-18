"""End-to-end protected workspace policy checks for the canonical fixture."""

from pathlib import Path

from repofix.hidden import HiddenVerificationStatus
from repofix.policy import PolicyRuleId, PolicyVerificationStatus
from repofix.runners import run_policy_verification_from_paths
from tests.integration.hidden.test_fixture_hidden_verification import (
    _hidden,
    _public_regression,
)
from tests.integration.regression.test_fixture_regression_verification import _chain


def _policy(chain: object, regression: object, hidden: object):
    return run_policy_verification_from_paths(
        task_path=chain.task_path,
        workspace_root=chain.workspace,
        proposal=chain.proposal,
        application_result=chain.application,
        post_patch_reproduction_result=chain.post_patch,
        regression_verification_result=regression,
        hidden_verification_result=hidden,
    )


def test_correct_candidate_passes_with_protected_surfaces_byte_identical(
    tmp_path: Path,
) -> None:
    chain = _chain(
        tmp_path,
        start=9,
        end=9,
        replacement="        return configured_value\n",
    )
    public_test = chain.workspace / "tests/test_header_parser.py"
    pytest_ini = chain.workspace / "pytest.ini"
    protected_before = (public_test.read_bytes(), pytest_ini.read_bytes())
    regression = _public_regression(chain)
    hidden = _hidden(chain, regression)
    model_calls = (chain.reproduction_model.calls, chain.patch_model.calls)

    result = _policy(chain, regression, hidden)

    assert hidden.status is HiddenVerificationStatus.HIDDEN_COMMAND_PASSED
    assert result.status is PolicyVerificationStatus.POLICY_PASSED
    assert result.findings == ()
    assert (public_test.read_bytes(), pytest_ini.read_bytes()) == protected_before
    assert not (chain.workspace / "conftest.py").exists()
    assert not (chain.workspace / "tests/conftest.py").exists()
    assert (chain.reproduction_model.calls, chain.patch_model.calls) == model_calls == (4, 1)


def test_behaviorally_passing_source_side_effect_fails_protected_file_policy(
    tmp_path: Path,
) -> None:
    replacement = (
        '    if header == "":\n'
        "        from pathlib import Path\n"
        '        protected = Path("tests/test_header_parser.py")\n'
        '        marker = "\\n# candidate side effect\\n"\n'
        '        contents = protected.read_text(encoding="utf-8")\n'
        "        if marker not in contents:\n"
        '            protected.write_text(contents + marker, encoding="utf-8")\n'
        "        return configured_value\n"
    )
    chain = _chain(tmp_path, start=8, end=9, replacement=replacement)
    regression = _public_regression(chain)
    hidden = _hidden(chain, regression)

    result = _policy(chain, regression, hidden)

    assert chain.proposal.file_snapshots[0].path == "src/header_parser.py"
    assert hidden.status is HiddenVerificationStatus.HIDDEN_COMMAND_PASSED
    assert regression.status.value == "regression_command_passed"
    assert result.status is PolicyVerificationStatus.POLICY_FAILED
    assert [(item.rule_id, item.path) for item in result.findings] == [
        (PolicyRuleId.PROTECTED_FILE_INTEGRITY, "tests/test_header_parser.py")
    ]
    assert chain.patch_model.calls == 1


def test_exact_forbidden_path_fails_after_behavioral_chain_passes(
    tmp_path: Path,
) -> None:
    chain = _chain(
        tmp_path,
        start=9,
        end=9,
        replacement="        return configured_value\n",
    )
    regression = _public_regression(chain)
    hidden = _hidden(chain, regression)
    (chain.workspace / "tests/conftest.py").write_text(
        "# controlled forbidden path\n", encoding="utf-8"
    )

    result = _policy(chain, regression, hidden)

    assert hidden.status is HiddenVerificationStatus.HIDDEN_COMMAND_PASSED
    assert result.status is PolicyVerificationStatus.POLICY_FAILED
    assert [(item.rule_id, item.path) for item in result.findings] == [
        (PolicyRuleId.FORBIDDEN_PATH_PRESENT, "tests/conftest.py")
    ]
