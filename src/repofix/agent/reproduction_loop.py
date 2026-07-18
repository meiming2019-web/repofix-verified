"""Provider-independent agent-requested reproduction workflow."""

import hashlib
import json
import re
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, model_validator

from repofix.agent.actions import (
    ListFilesAction,
    ReadFileAction,
    RecordHypothesisAction,
    RunApprovedCommandAction,
    SearchCodeAction,
    UnderstandIssueAction,
)
from repofix.agent.interfaces import AgentModel, ReadOnlyToolGateway
from repofix.agent.loop import (
    AgentProtocolError,
    _execute_tool_action,
    _invalid_action,
)
from repofix.agent.state import (
    AgentPhase,
    AgentReproductionObservation,
    AgentState,
    AgentWorkflow,
    REPRODUCED_TERMINAL_SUMMARY,
)
from repofix.execution import ApprovedCommandExecutionError, ApprovedCommandExecutionResult
from repofix.reproduction import (
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionStatus,
    ReproductionVerdict,
    compute_reproduction_expectation_fingerprint,
    verify_reproduction,
)
from repofix.tasks import AgentTaskSpec


ReadOnlyToolAction = ListFilesAction | SearchCodeAction | ReadFileAction
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def compute_task_fingerprint(task: AgentTaskSpec) -> str:
    """Hash the complete agent task using deterministic canonical JSON."""
    canonical = json.dumps(
        task.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validated_state_update(state: AgentState, **updates: object) -> AgentState:
    values = state.model_dump()
    values.update(updates)
    return AgentState.model_validate(values)


class ApprovedCommandGateway(Protocol):
    """Exact-ID approved command execution boundary."""

    def execute(self, command_id: str) -> ApprovedCommandExecutionResult:
        """Execute one trusted command selected only by its identifier."""
        ...


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvaluatorReproductionAttempt(_StrictFrozenModel):
    """Evaluator-only evidence and full verdict for one command attempt."""

    evidence: ReproductionEvidence
    verdict: ReproductionVerdict

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.evidence.command_id != self.verdict.command_id:
            raise ValueError("attempt evidence and verdict command IDs must match")
        if self.evidence.exit_code != self.verdict.exit_code:
            raise ValueError("attempt evidence and verdict exit codes must match")
        return self


class ReproductionAgentRunResult(_StrictFrozenModel):
    """Public agent state plus evaluator-only reproduction audit records."""

    state: AgentState
    attempts: tuple[EvaluatorReproductionAttempt, ...]
    task_fingerprint: str
    reproduction_expectation_fingerprint: str

    @model_validator(mode="before")
    @classmethod
    def validate_fingerprint_shape(cls, values: object) -> object:
        if isinstance(values, dict):
            for field_name, description in (
                ("task_fingerprint", "task fingerprint"),
                (
                    "reproduction_expectation_fingerprint",
                    "reproduction expectation fingerprint",
                ),
            ):
                value = values.get(field_name)
                if not isinstance(value, str) or not _FINGERPRINT_PATTERN.fullmatch(value):
                    raise ValueError(f"{description} must be lowercase hexadecimal SHA-256")
        return values

    @model_validator(mode="after")
    def validate_attempt_observations(self) -> Self:
        if self.state.workflow is not AgentWorkflow.REPRODUCTION:
            raise ValueError("reproduction run results require reproduction workflow state")
        observations = self.state.reproduction_observations
        if len(self.attempts) > 1:
            raise ValueError("reproduction run results permit at most one attempt")
        if (
            any(
                attempt.verdict.status is ReproductionStatus.REPRODUCED for attempt in self.attempts
            )
            and self.state.phase is not AgentPhase.FINISHED
        ):
            raise ValueError("reproduced attempts require finished reproduction state")
        if len(observations) != len(self.attempts):
            raise ValueError("each reproduction attempt requires one public observation")
        for observation, attempt in zip(observations, self.attempts, strict=True):
            evidence = attempt.evidence
            if (
                observation.command_id != self.state.reproduction_command_id
                or evidence.command_id != self.state.reproduction_command_id
                or attempt.verdict.command_id != self.state.reproduction_command_id
                or observation.termination_reason.value != evidence.termination_reason.value
                or observation.exit_code != evidence.exit_code
                or observation.stdout != evidence.stdout
                or observation.stderr != evidence.stderr
                or observation.stdout_bytes != evidence.stdout_bytes
                or observation.stderr_bytes != evidence.stderr_bytes
                or observation.had_decode_errors != evidence.had_decode_errors
                or observation.status is not attempt.verdict.status
            ):
                raise ValueError("public reproduction observation does not match its attempt")
        if self.state.phase is AgentPhase.FINISHED and (
            len(self.attempts) != 1
            or self.attempts[0].verdict.status is not ReproductionStatus.REPRODUCED
        ):
            raise ValueError("finished reproduction results require exactly one reproduced attempt")
        return self


def compute_reproduction_run_fingerprint(result: ReproductionAgentRunResult) -> str:
    """Hash one complete reproduction run using deterministic canonical JSON."""
    canonical = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _public_observation(
    result: ApprovedCommandExecutionResult, verdict: ReproductionVerdict
) -> AgentReproductionObservation:
    return AgentReproductionObservation(
        command_id=result.command_id,
        termination_reason=result.termination_reason,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes,
        had_decode_errors=result.had_decode_errors,
        status=verdict.status,
    )


def _validate_limit(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a strict positive integer")
    return value


def run_reproduction_agent_loop(
    *,
    task: AgentTaskSpec,
    expectation: ReproductionExpectation,
    model: AgentModel,
    tools: ReadOnlyToolGateway,
    command_gateway: ApprovedCommandGateway,
    max_steps: int,
) -> ReproductionAgentRunResult:
    """Run bounded investigation followed by evaluator-controlled reproduction."""
    _validate_limit(max_steps, name="max_steps")
    if expectation.command_id not in task.approved_commands:
        raise AgentProtocolError("reproduction expectation command ID is not approved")

    state = AgentState.initial(
        task.task_id,
        workflow=AgentWorkflow.REPRODUCTION,
        reproduction_command_id=expectation.command_id,
    )
    attempts: list[EvaluatorReproductionAttempt] = []

    for _ in range(max_steps):
        action = model.next_action(task=task, state=state)
        next_step_count = state.step_count + 1

        if state.phase is AgentPhase.UNDERSTAND:
            if not isinstance(action, UnderstandIssueAction):
                raise _invalid_action(action, state.phase)
            state = _validated_state_update(
                state,
                phase=AgentPhase.EXPLORE,
                issue_understanding=action.understanding,
                step_count=next_step_count,
            )
            continue

        if isinstance(action, (ListFilesAction, SearchCodeAction, ReadFileAction)):
            if state.phase not in {AgentPhase.EXPLORE, AgentPhase.HYPOTHESIZE}:
                raise _invalid_action(action, state.phase)
            observation = _execute_tool_action(
                action=action,
                tools=tools,
                step_index=state.step_count,
            )
            state = _validated_state_update(
                state,
                phase=AgentPhase.EXPLORE,
                observations=(*state.observations, observation),
                step_count=next_step_count,
            )
            continue

        if isinstance(action, RecordHypothesisAction):
            if state.phase not in {AgentPhase.EXPLORE, AgentPhase.HYPOTHESIZE}:
                raise _invalid_action(action, state.phase)
            state = _validated_state_update(
                state,
                phase=AgentPhase.HYPOTHESIZE,
                hypotheses=(*state.hypotheses, action.hypothesis),
                step_count=next_step_count,
            )
            continue

        if isinstance(action, RunApprovedCommandAction):
            if state.phase is not AgentPhase.HYPOTHESIZE:
                raise _invalid_action(action, state.phase)
            if not any(hypothesis.status == "supported" for hypothesis in state.hypotheses):
                raise AgentProtocolError(
                    "approved commands require at least one supported hypothesis"
                )
            if not any(observation.success for observation in state.observations):
                raise AgentProtocolError(
                    "approved commands require at least one successful repository observation"
                )
            if action.command_id != state.reproduction_command_id:
                raise AgentProtocolError("requested command ID is not configured for reproduction")
            if attempts:
                raise AgentProtocolError("an approved command may execute only once per run")

            execution_result = command_gateway.execute(action.command_id)
            approved_argv = task.approved_commands[action.command_id].argv
            if (
                execution_result.command_id != action.command_id
                or execution_result.argv != approved_argv
            ):
                raise ApprovedCommandExecutionError(
                    "approved command gateway returned inconsistent execution identity"
                )
            evidence = ReproductionEvidence.from_execution_result(execution_result)
            verdict = verify_reproduction(expectation=expectation, evidence=evidence)
            attempt = EvaluatorReproductionAttempt(evidence=evidence, verdict=verdict)
            reproduction_observation = _public_observation(execution_result, verdict)
            attempts.append(attempt)
            if verdict.status is ReproductionStatus.REPRODUCED:
                state = _validated_state_update(
                    state,
                    phase=AgentPhase.FINISHED,
                    reproduction_observations=(reproduction_observation,),
                    step_count=next_step_count,
                    terminal_summary=REPRODUCED_TERMINAL_SUMMARY,
                )
                return ReproductionAgentRunResult(
                    state=state,
                    attempts=tuple(attempts),
                    task_fingerprint=compute_task_fingerprint(task),
                    reproduction_expectation_fingerprint=(
                        compute_reproduction_expectation_fingerprint(expectation)
                    ),
                )
            state = _validated_state_update(
                state,
                phase=AgentPhase.EXPLORE,
                reproduction_observations=(
                    *state.reproduction_observations,
                    reproduction_observation,
                ),
                step_count=next_step_count,
            )
            continue

        raise _invalid_action(action, state.phase)

    state = _validated_state_update(
        state,
        phase=AgentPhase.FAILED,
        failure_reason=f"reproduction workflow exceeded the {max_steps}-step budget",
    )
    return ReproductionAgentRunResult(
        state=state,
        attempts=tuple(attempts),
        task_fingerprint=compute_task_fingerprint(task),
        reproduction_expectation_fingerprint=(
            compute_reproduction_expectation_fingerprint(expectation)
        ),
    )
