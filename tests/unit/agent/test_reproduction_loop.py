"""Tests for the provider-independent agent-requested reproduction loop."""

import hashlib
from typing import Any

import pytest

import repofix.agent.reproduction_loop as reproduction_loop_module

from repofix.agent import (
    AgentAction,
    AgentPhase,
    AgentProtocolError,
    AgentState,
    AgentWorkflow,
    FinishInvestigationAction,
    IssueUnderstanding,
    ReadFileResult,
    RecordHypothesisAction,
    RepairHypothesis,
    RunApprovedCommandAction,
    SearchCodeAction,
    UnderstandIssueAction,
    run_read_only_investigation,
    run_reproduction_agent_loop,
)
from repofix.agent.state import REPRODUCED_TERMINAL_SUMMARY
from repofix.execution import (
    ApprovedCommandExecutionError,
    ApprovedCommandExecutionResult,
    CommandTerminationReason,
)
from repofix.reproduction import (
    ReproductionExpectation,
    ReproductionOutputFragment,
    ReproductionOutputStream,
    ReproductionStatus,
)
from repofix.tasks import AgentTaskSpec


def task_data() -> dict[str, Any]:
    return {
        "task_id": "workflow-task",
        "repository_url": "https://github.com/example/project.git",
        "pre_fix_commit": "0123456789abcdef0123456789abcdef01234567",
        "issue_title": "Target behavior fails",
        "issue_body": "The target behavior produces an incorrect result.",
        "approved_commands": {
            "unit_tests": {"argv": ["pytest", "-q"]},
            "other_tests": {"argv": ["pytest", "tests/other"]},
        },
        "allowed_source_paths": ["src", "tests"],
        "timeout_seconds": 300,
    }


def task() -> AgentTaskSpec:
    return AgentTaskSpec.model_validate(task_data())


def expectation(command_id: str = "unit_tests") -> ReproductionExpectation:
    return ReproductionExpectation(
        command_id=command_id,
        expected_exit_codes=(1,),
        required_fragments=(
            ReproductionOutputFragment(
                fragment_id="target-signature",
                stream=ReproductionOutputStream.COMBINED,
                text="TARGET FAILURE",
            ),
        ),
        forbidden_fragments=(
            ReproductionOutputFragment(
                fragment_id="collection-error",
                stream=ReproductionOutputStream.COMBINED,
                text="ERROR collecting",
            ),
        ),
    )


def understanding_action() -> UnderstandIssueAction:
    return UnderstandIssueAction(
        kind="understand_issue",
        understanding=IssueUnderstanding(
            expected_behavior="The target behavior succeeds.",
            observed_behavior="The target behavior fails.",
            reproduction_clues=("The issue identifies the failing case.",),
            likely_components=("src/parser.py",),
            missing_information=(),
        ),
    )


def hypothesis_action(
    *, status: str = "supported", identifier: str = "target-hypothesis"
) -> RecordHypothesisAction:
    return RecordHypothesisAction(
        kind="record_hypothesis",
        hypothesis=RepairHypothesis.model_validate(
            {
                "hypothesis_id": identifier,
                "description": "The target branch returns the wrong value.",
                "supporting_evidence": ["Repository evidence identifies the branch."],
                "contradicting_evidence": [],
                "confidence": 0.9,
                "status": status,
            }
        ),
    )


def execution_result(
    *,
    command_id: str = "unit_tests",
    stdout: str = "TARGET FAILURE\n",
    exit_code: int | None = 1,
    reason: CommandTerminationReason = CommandTerminationReason.COMPLETED,
    argv: tuple[str, ...] = ("pytest", "-q"),
) -> ApprovedCommandExecutionResult:
    return ApprovedCommandExecutionResult(
        command_id=command_id,
        argv=argv,
        termination_reason=reason,
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
        stdout_bytes=len(stdout.encode("utf-8")),
        stderr_bytes=0,
        had_decode_errors=False,
    )


class ScriptedModel:
    def __init__(self, actions: list[AgentAction]) -> None:
        self.actions = actions
        self.states: list[AgentState] = []

    def next_action(self, *, task: AgentTaskSpec, state: AgentState) -> AgentAction:
        self.states.append(state)
        return self.actions[len(self.states) - 1]


class FakeTools:
    def list_files(self, path: str) -> str:
        return "src/parser.py"

    def search_code(self, query: str, file_glob: str | None = None) -> str:
        return "src/parser.py:1:def parse_target"

    def read_file(self, path: str, start_line: int, end_line: int) -> str:
        return "1: def parse_target():\n2:     return WRONG"

    def read_file_with_metadata(self, path: str, start_line: int, end_line: int) -> ReadFileResult:
        return ReadFileResult(
            output=self.read_file(path, start_line, end_line),
            full_file_sha256=hashlib.sha256(b"complete fake source").hexdigest(),
        )


class FakeCommandGateway:
    def __init__(
        self,
        results: list[ApprovedCommandExecutionResult] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[str] = []

    def execute(self, command_id: str) -> ApprovedCommandExecutionResult:
        self.calls.append(command_id)
        if self.error is not None:
            raise self.error
        return self.results[len(self.calls) - 1]


def happy_actions() -> list[AgentAction]:
    return [
        understanding_action(),
        SearchCodeAction(kind="search_code", query="parse_target", file_glob="*.py"),
        hypothesis_action(),
        RunApprovedCommandAction(command_id="unit_tests"),
    ]


def run_script(
    actions: list[AgentAction],
    gateway: FakeCommandGateway,
    *,
    max_steps: int | object | None = None,
    expected: ReproductionExpectation | None = None,
):
    return run_reproduction_agent_loop(
        task=task(),
        expectation=expected or expectation(),
        model=ScriptedModel(actions),
        tools=FakeTools(),
        command_gateway=gateway,
        max_steps=len(actions) if max_steps is None else max_steps,  # type: ignore[arg-type]
    )


def test_happy_path_reaches_finished_with_exact_id_and_private_attempt() -> None:
    gateway = FakeCommandGateway([execution_result()])

    result = run_script(happy_actions(), gateway)

    assert result.state.workflow is AgentWorkflow.REPRODUCTION
    assert result.state.phase is AgentPhase.FINISHED
    assert result.state.reproduction_command_id == "unit_tests"
    assert result.state.terminal_summary == REPRODUCED_TERMINAL_SUMMARY
    assert result.state.step_count == 4
    assert gateway.calls == ["unit_tests"]
    assert len(result.attempts) == 1
    assert result.attempts[0].verdict.status is ReproductionStatus.REPRODUCED
    assert result.state.reproduction_observations[0].status is ReproductionStatus.REPRODUCED
    public_rendered = repr(result.state.model_dump())
    assert "target-signature" not in public_rendered
    assert "collection-error" not in public_rendered
    assert "expected_exit_codes" not in public_rendered
    assert "TARGET FAILURE" in public_rendered


def test_run_result_rejects_multiple_attempts_and_finished_without_attempt() -> None:
    result = run_script(happy_actions(), FakeCommandGateway([execution_result()]))
    attempt = result.attempts[0]

    with pytest.raises(ValueError, match="at most one attempt"):
        result.model_validate(
            {
                "state": result.state,
                "attempts": (attempt, attempt),
                "task_fingerprint": result.task_fingerprint,
                "reproduction_expectation_fingerprint": (
                    result.reproduction_expectation_fingerprint
                ),
            }
        )

    with pytest.raises(ValueError, match="reproduction expectation fingerprint"):
        result.model_validate(
            {
                **result.model_dump(),
                "reproduction_expectation_fingerprint": "A" * 64,
            }
        )
    with pytest.raises(ValueError, match="one public observation"):
        result.model_validate(
            {
                "state": result.state,
                "attempts": (),
                "task_fingerprint": result.task_fingerprint,
                "reproduction_expectation_fingerprint": (
                    result.reproduction_expectation_fingerprint
                ),
            }
        )


def test_failed_run_result_rejects_reproduced_attempt() -> None:
    successful = run_script(happy_actions(), FakeCommandGateway([execution_result()]))
    failed_state = AgentState(
        task_id="workflow-task",
        phase=AgentPhase.FAILED,
        issue_understanding=None,
        hypotheses=(),
        observations=(),
        step_count=4,
        terminal_summary=None,
        failure_reason="Step budget exhausted.",
        workflow=AgentWorkflow.REPRODUCTION,
        reproduction_command_id="unit_tests",
        reproduction_observations=(),
    )

    with pytest.raises(ValueError, match="reproduced attempts require finished"):
        successful.model_validate(
            {
                "state": failed_state,
                "attempts": successful.attempts,
                "task_fingerprint": successful.task_fingerprint,
                "reproduction_expectation_fingerprint": (
                    successful.reproduction_expectation_fingerprint
                ),
            }
        )


def test_reproduction_on_last_step_finishes_without_another_model_call() -> None:
    model = ScriptedModel(happy_actions())
    gateway = FakeCommandGateway([execution_result()])

    result = run_reproduction_agent_loop(
        task=task(),
        expectation=expectation(),
        model=model,
        tools=FakeTools(),
        command_gateway=gateway,
        max_steps=4,
    )

    assert len(model.states) == 4
    assert model.states[-1].phase is AgentPhase.HYPOTHESIZE
    assert result.state.phase is AgentPhase.FINISHED
    assert result.state.step_count == 4
    assert result.state.terminal_summary == REPRODUCED_TERMINAL_SUMMARY


def test_finish_is_rejected_before_reproduction() -> None:
    actions = [
        understanding_action(),
        hypothesis_action(),
        FinishInvestigationAction(kind="finish_investigation", summary="Done"),
    ]

    with pytest.raises(AgentProtocolError, match="not permitted"):
        run_script(actions, FakeCommandGateway())


@pytest.mark.parametrize("status", ["unverified", "rejected"])
def test_command_requires_supported_hypothesis(status: str) -> None:
    actions = [
        understanding_action(),
        hypothesis_action(status=status),
        RunApprovedCommandAction(command_id="unit_tests"),
    ]

    with pytest.raises(AgentProtocolError, match="supported hypothesis"):
        run_script(actions, FakeCommandGateway())


def test_command_requires_successful_repository_observation() -> None:
    actions = [
        understanding_action(),
        hypothesis_action(),
        RunApprovedCommandAction(command_id="unit_tests"),
    ]

    with pytest.raises(AgentProtocolError, match="successful repository observation"):
        run_script(actions, FakeCommandGateway())


def test_failed_repository_observation_does_not_enable_command() -> None:
    class FailedTools(FakeTools):
        def search_code(self, query: str, file_glob: str | None = None) -> str:
            from repofix.agent import ToolExecutionError

            raise ToolExecutionError("sanitized search failure")

    model = ScriptedModel(
        [
            understanding_action(),
            SearchCodeAction(kind="search_code", query="target"),
            hypothesis_action(),
            RunApprovedCommandAction(command_id="unit_tests"),
        ]
    )
    gateway = FakeCommandGateway()

    with pytest.raises(AgentProtocolError, match="successful repository observation"):
        run_reproduction_agent_loop(
            task=task(),
            expectation=expectation(),
            model=model,
            tools=FailedTools(),
            command_gateway=gateway,
            max_steps=4,
        )

    assert model.states[-1].observations[-1].success is False
    assert gateway.calls == []


def test_reproduction_action_is_rejected_by_investigation_loop() -> None:
    model = ScriptedModel([RunApprovedCommandAction(command_id="unit_tests")])

    with pytest.raises(AgentProtocolError, match="not permitted"):
        run_read_only_investigation(task=task(), model=model, tools=FakeTools(), max_steps=1)


def test_unknown_command_is_rejected_before_gateway() -> None:
    gateway = FakeCommandGateway()
    actions = [
        understanding_action(),
        SearchCodeAction(kind="search_code", query="target"),
        hypothesis_action(),
        RunApprovedCommandAction(command_id="missing_tests"),
    ]

    with pytest.raises(AgentProtocolError, match="not configured"):
        run_script(actions, gateway)

    assert gateway.calls == []


def test_duplicate_command_execution_is_rejected() -> None:
    gateway = FakeCommandGateway([execution_result(exit_code=0, stdout="")])
    actions = [
        understanding_action(),
        SearchCodeAction(kind="search_code", query="target"),
        hypothesis_action(),
        RunApprovedCommandAction(command_id="unit_tests"),
        hypothesis_action(identifier="revised-hypothesis"),
        RunApprovedCommandAction(command_id="unit_tests"),
    ]

    with pytest.raises(AgentProtocolError, match="only once"):
        run_script(actions, gateway)

    assert gateway.calls == ["unit_tests"]


def test_other_approved_command_is_rejected_before_second_gateway_call() -> None:
    gateway = FakeCommandGateway([execution_result(exit_code=0, stdout="")])
    actions = [
        understanding_action(),
        SearchCodeAction(kind="search_code", query="target"),
        hypothesis_action(),
        RunApprovedCommandAction(command_id="unit_tests"),
        hypothesis_action(identifier="revised-hypothesis"),
        RunApprovedCommandAction(command_id="other_tests"),
    ]

    with pytest.raises(AgentProtocolError, match="not configured"):
        run_script(actions, gateway)

    assert gateway.calls == ["unit_tests"]


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (execution_result(exit_code=0, stdout=""), ReproductionStatus.NOT_REPRODUCED),
        (execution_result(stdout="unrelated failure"), ReproductionStatus.INCONCLUSIVE),
        (
            execution_result(
                stdout="partial",
                exit_code=None,
                reason=CommandTerminationReason.TIMED_OUT,
            ),
            ReproductionStatus.INCONCLUSIVE,
        ),
        (
            execution_result(
                stdout="partial",
                exit_code=None,
                reason=CommandTerminationReason.OUTPUT_LIMIT,
            ),
            ReproductionStatus.INCONCLUSIVE,
        ),
    ],
    ids=["not-reproduced", "unrelated", "timeout", "output-limit"],
)
def test_nonreproduced_results_return_to_explore(
    result: ApprovedCommandExecutionResult, status: ReproductionStatus
) -> None:
    model = ScriptedModel(
        [
            understanding_action(),
            SearchCodeAction(kind="search_code", query="target"),
            hypothesis_action(),
            RunApprovedCommandAction(command_id="unit_tests"),
            hypothesis_action(identifier="next-hypothesis"),
        ]
    )

    final = run_reproduction_agent_loop(
        task=task(),
        expectation=expectation(),
        model=model,
        tools=FakeTools(),
        command_gateway=FakeCommandGateway([result]),
        max_steps=5,
    )

    assert model.states[-1].phase is AgentPhase.EXPLORE
    assert model.states[-1].reproduction_observations[-1].status is status
    assert final.state.phase is AgentPhase.FAILED


def test_exit_zero_and_unrelated_failure_do_not_permit_finish() -> None:
    for result in (
        execution_result(exit_code=0, stdout=""),
        execution_result(stdout="unrelated assertion failure"),
    ):
        actions = [
            understanding_action(),
            SearchCodeAction(kind="search_code", query="target"),
            hypothesis_action(),
            RunApprovedCommandAction(command_id="unit_tests"),
            FinishInvestigationAction(
                kind="finish_investigation",
                summary="The reported behavior was reproduced.",
            ),
        ]
        with pytest.raises(AgentProtocolError, match="not permitted"):
            run_script(actions, FakeCommandGateway([result]))


@pytest.mark.parametrize(
    "error",
    [
        ApprovedCommandExecutionError("executor failed"),
        TypeError("programmer failure"),
        AssertionError("invariant failure"),
    ],
)
def test_executor_and_programmer_errors_propagate_without_retry(error: BaseException) -> None:
    gateway = FakeCommandGateway(error=error)
    actions = [
        understanding_action(),
        SearchCodeAction(kind="search_code", query="target"),
        hypothesis_action(),
        RunApprovedCommandAction(command_id="unit_tests"),
    ]

    with pytest.raises(type(error)):
        run_script(actions, gateway)

    assert gateway.calls == ["unit_tests"]


@pytest.mark.parametrize(
    "result",
    [
        execution_result(command_id="unit_tests", argv=("pytest", "tests/other")),
        execution_result(command_id="other_tests", argv=("pytest", "-q")),
    ],
    ids=["mismatched-command-id", "mismatched-argv"],
)
def test_gateway_identity_mismatch_is_rejected_before_verification(
    monkeypatch: pytest.MonkeyPatch,
    result: ApprovedCommandExecutionResult,
) -> None:
    gateway = FakeCommandGateway([result])
    actions = [
        understanding_action(),
        SearchCodeAction(kind="search_code", query="target"),
        hypothesis_action(),
        RunApprovedCommandAction(command_id="other_tests"),
    ]
    verifier_called = False

    def fail_verifier(**kwargs: object) -> object:
        nonlocal verifier_called
        verifier_called = True
        raise AssertionError("verifier must not run")

    monkeypatch.setattr(reproduction_loop_module, "verify_reproduction", fail_verifier)

    with pytest.raises(
        ApprovedCommandExecutionError,
        match="gateway returned inconsistent execution identity",
    ) as caught:
        run_script(actions, gateway, expected=expectation("other_tests"))

    assert gateway.calls == ["other_tests"]
    assert verifier_called is False
    assert "pytest" not in str(caught.value)


def test_step_budget_exhaustion_returns_failed_without_extra_model_call() -> None:
    model = ScriptedModel(
        [understanding_action(), SearchCodeAction(kind="search_code", query="target")]
    )

    result = run_reproduction_agent_loop(
        task=task(),
        expectation=expectation(),
        model=model,
        tools=FakeTools(),
        command_gateway=FakeCommandGateway(),
        max_steps=2,
    )

    assert result.state.phase is AgentPhase.FAILED
    assert result.state.step_count == 2
    assert len(model.states) == 2


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_rejects_invalid_strict_limits(value: object) -> None:
    with pytest.raises(ValueError, match="strict positive integer"):
        run_script([], FakeCommandGateway(), max_steps=value)


def test_multi_attempt_api_is_absent() -> None:
    with pytest.raises(TypeError, match="max_reproduction_attempts"):
        run_reproduction_agent_loop(  # type: ignore[call-arg]
            task=task(),
            expectation=expectation(),
            model=ScriptedModel([]),
            tools=FakeTools(),
            command_gateway=FakeCommandGateway(),
            max_steps=1,
            max_reproduction_attempts=2,
        )


def test_expectation_command_must_exist_before_model_or_gateway_call() -> None:
    gateway = FakeCommandGateway()
    model = ScriptedModel([])

    with pytest.raises(AgentProtocolError, match="expectation command ID"):
        run_reproduction_agent_loop(
            task=task(),
            expectation=expectation("missing_tests"),
            model=model,
            tools=FakeTools(),
            command_gateway=gateway,
            max_steps=1,
        )

    assert model.states == []
    assert gateway.calls == []
