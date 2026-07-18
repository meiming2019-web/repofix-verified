"""Tests for deterministic post-patch reproduction verification."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from repofix.agent import (
    AgentPhase,
    AgentReproductionObservation,
    AgentState,
    AgentWorkflow,
    EvaluatorReproductionAttempt,
    IssueUnderstanding,
    RepairHypothesis,
    ReproductionAgentRunResult,
    ToolObservation,
)
from repofix.agent.reproduction_loop import compute_task_fingerprint
from repofix.agent.state import REPRODUCED_TERMINAL_SUMMARY
from repofix.execution import (
    ApprovedCommandExecutionError,
    ApprovedCommandExecutionResult,
    CommandTerminationReason,
)
from repofix.patching import (
    PatchApplicationResult,
    PatchProposalDraft,
    apply_validated_patch_proposal,
    validate_patch_proposal,
)
from repofix.reproduction import (
    POST_PATCH_INCONCLUSIVE_SUMMARY,
    POST_PATCH_NOT_REPRODUCED_SUMMARY,
    POST_PATCH_STILL_REPRODUCED_SUMMARY,
    PostPatchReproductionError,
    PostPatchReproductionResult,
    PostPatchReproductionStatus,
    ReproductionEvidence,
    ReproductionExpectation,
    ReproductionStatus,
    ReproductionTerminationReason,
    ReproductionVerdict,
    compute_reproduction_expectation_fingerprint,
    verify_post_patch_reproduction,
)
from repofix.tasks import AgentTaskSpec


@dataclass
class Prepared:
    workspace: Path
    source: Path
    task: AgentTaskSpec
    expectation: ReproductionExpectation
    reproduction_result: ReproductionAgentRunResult
    proposal: object
    application_result: PatchApplicationResult


class Gateway:
    def __init__(
        self,
        result: ApprovedCommandExecutionResult | None = None,
        *,
        error: ApprovedCommandExecutionError | None = None,
        mutation: tuple[Path, bytes] | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.mutation = mutation
        self.calls: list[str] = []

    def execute(self, command_id: str) -> ApprovedCommandExecutionResult:
        self.calls.append(command_id)
        if self.mutation is not None:
            path, contents = self.mutation
            path.write_bytes(contents)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _execution(
    *,
    status: ReproductionStatus,
    termination: CommandTerminationReason | None = None,
) -> ApprovedCommandExecutionResult:
    if status is ReproductionStatus.REPRODUCED:
        stdout, exit_code = "TARGET FAILURE\n", 1
    elif status is ReproductionStatus.NOT_REPRODUCED:
        stdout, exit_code = "1 passed\n", 0
    else:
        stdout, exit_code = "partial\n", None
    return ApprovedCommandExecutionResult(
        command_id="unit_tests",
        argv=("pytest", "-q"),
        termination_reason=termination
        or (
            CommandTerminationReason.TIMED_OUT
            if status is ReproductionStatus.INCONCLUSIVE
            else CommandTerminationReason.COMPLETED
        ),
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
        stdout_bytes=len(stdout.encode("utf-8")),
        stderr_bytes=0,
        had_decode_errors=False,
    )


def _original_result(
    task: AgentTaskSpec, expectation: ReproductionExpectation, source: Path
) -> ReproductionAgentRunResult:
    output = "TARGET FAILURE\n"
    evidence = ReproductionEvidence(
        command_id="unit_tests",
        argv=("pytest", "-q"),
        termination_reason=ReproductionTerminationReason.COMPLETED,
        exit_code=1,
        stdout=output,
        stderr="",
        stdout_bytes=len(output.encode("utf-8")),
        stderr_bytes=0,
        had_decode_errors=False,
    )
    verdict = ReproductionVerdict(
        status=ReproductionStatus.REPRODUCED,
        command_id="unit_tests",
        exit_code=1,
        reasons=("expected failing behavior was reproduced",),
        matched_required_fragment_ids=("target",),
        missing_required_fragment_ids=(),
        forbidden_fragment_ids_found=(),
    )
    public = AgentReproductionObservation(
        command_id="unit_tests",
        termination_reason=CommandTerminationReason.COMPLETED,
        exit_code=1,
        stdout=output,
        stderr="",
        stdout_bytes=len(output.encode("utf-8")),
        stderr_bytes=0,
        had_decode_errors=False,
        status=ReproductionStatus.REPRODUCED,
    )
    state = AgentState(
        task_id=task.task_id,
        phase=AgentPhase.FINISHED,
        issue_understanding=IssueUnderstanding(
            expected_behavior="right",
            observed_behavior="wrong",
            reproduction_clues=("target",),
            likely_components=("src/app.py",),
            missing_information=(),
        ),
        hypotheses=(
            RepairHypothesis(
                hypothesis_id="h1",
                description="wrong return",
                supporting_evidence=("source read",),
                contradicting_evidence=(),
                confidence=0.9,
                status="supported",
            ),
        ),
        observations=(
            ToolObservation(
                step_index=1,
                tool_name="read_file",
                arguments={"path": "src/app.py", "start_line": 1, "end_line": 2},
                success=True,
                output="1: def value():\n2:     return 'wrong'\n",
                error=None,
                full_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            ),
        ),
        step_count=4,
        terminal_summary=REPRODUCED_TERMINAL_SUMMARY,
        failure_reason=None,
        workflow=AgentWorkflow.REPRODUCTION,
        reproduction_command_id="unit_tests",
        reproduction_observations=(public,),
    )
    return ReproductionAgentRunResult(
        state=state,
        attempts=(EvaluatorReproductionAttempt(evidence=evidence, verdict=verdict),),
        task_fingerprint=compute_task_fingerprint(task),
        reproduction_expectation_fingerprint=(
            compute_reproduction_expectation_fingerprint(expectation)
        ),
    )


@pytest.fixture
def prepared(tmp_path: Path) -> Prepared:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/app.py"
    source.write_bytes(b"def value():\n    return 'wrong'\n")
    task = AgentTaskSpec.model_validate(
        {
            "task_id": "post-patch-task",
            "repository_url": "https://github.com/example/project.git",
            "pre_fix_commit": "0" * 40,
            "issue_title": "Wrong return",
            "issue_body": "The target behavior fails.",
            "approved_commands": {"unit_tests": {"argv": ["pytest", "-q"]}},
            "allowed_source_paths": ["src"],
            "patchable_source_paths": ["src"],
            "timeout_seconds": 30,
        }
    )
    expectation = ReproductionExpectation.model_validate(
        {
            "command_id": "unit_tests",
            "expected_exit_codes": [1],
            "required_fragments": [
                {"fragment_id": "target", "stream": "combined", "text": "TARGET FAILURE"}
            ],
        }
    )
    reproduction_result = _original_result(task, expectation, source)
    proposal = validate_patch_proposal(
        workspace_root=tmp_path,
        task=task,
        reproduction_result=reproduction_result,
        draft=PatchProposalDraft.model_validate(
            {
                "hypothesis_id": "h1",
                "model_summary": "bounded change",
                "edits": [
                    {
                        "path": "src/app.py",
                        "start_line": 2,
                        "end_line": 2,
                        "replacement_text": "    return 'right'\n",
                        "rationale": "Correct the return.",
                    }
                ],
            }
        ),
    )
    application_result = apply_validated_patch_proposal(
        workspace_root=tmp_path,
        task=task,
        reproduction_result=reproduction_result,
        proposal=proposal,
    )
    return Prepared(
        workspace=tmp_path,
        source=source,
        task=task,
        expectation=expectation,
        reproduction_result=reproduction_result,
        proposal=proposal,
        application_result=application_result,
    )


def _verify(prepared: Prepared, gateway: Gateway, **updates: object):
    values = {
        "workspace_root": prepared.workspace,
        "task": prepared.task,
        "expectation": prepared.expectation,
        "original_reproduction_result": prepared.reproduction_result,
        "proposal": prepared.proposal,
        "application_result": prepared.application_result,
        "command_gateway": gateway,
    }
    values.update(updates)
    return verify_post_patch_reproduction(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("verifier_status", "post_status", "summary"),
    [
        (
            ReproductionStatus.NOT_REPRODUCED,
            PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_NOT_REPRODUCED,
            POST_PATCH_NOT_REPRODUCED_SUMMARY,
        ),
        (
            ReproductionStatus.REPRODUCED,
            PostPatchReproductionStatus.ORIGINAL_BEHAVIOR_STILL_REPRODUCED,
            POST_PATCH_STILL_REPRODUCED_SUMMARY,
        ),
        (
            ReproductionStatus.INCONCLUSIVE,
            PostPatchReproductionStatus.INCONCLUSIVE,
            POST_PATCH_INCONCLUSIVE_SUMMARY,
        ),
    ],
)
def test_existing_verifier_status_maps_to_post_patch_meaning_and_summary(
    prepared: Prepared,
    verifier_status: ReproductionStatus,
    post_status: PostPatchReproductionStatus,
    summary: str,
) -> None:
    gateway = Gateway(_execution(status=verifier_status))
    before = prepared.source.read_bytes()

    result = _verify(prepared, gateway)

    assert result.verifier_verdict.status is verifier_status
    assert result.status is post_status
    assert result.verification_summary == summary
    assert gateway.calls == ["unit_tests"]
    assert prepared.source.read_bytes() == before
    rendered = result.verification_summary.lower()
    for claim in ("bug is fixed", "patch is correct", "regression-free", "hidden tests pass"):
        assert claim not in rendered
    with pytest.raises(ValidationError):
        result.status = PostPatchReproductionStatus.INCONCLUSIVE  # type: ignore[misc]
    with pytest.raises(ValidationError):
        PostPatchReproductionResult.model_validate({**result.model_dump(), "extra": True})


def test_output_limit_remains_inconclusive(prepared: Prepared) -> None:
    result = _verify(
        prepared,
        Gateway(
            _execution(
                status=ReproductionStatus.INCONCLUSIVE,
                termination=CommandTerminationReason.OUTPUT_LIMIT,
            )
        ),
    )

    assert result.status is PostPatchReproductionStatus.INCONCLUSIVE
    assert result.verifier_verdict.status is ReproductionStatus.INCONCLUSIVE
    assert result.evidence.termination_reason is ReproductionTerminationReason.OUTPUT_LIMIT


@pytest.mark.parametrize("mismatch", ["task", "expectation"])
def test_current_fingerprint_mismatch_is_rejected_before_execution(
    prepared: Prepared, mismatch: str
) -> None:
    gateway = Gateway(_execution(status=ReproductionStatus.NOT_REPRODUCED))
    updates: dict[str, object]
    if mismatch == "task":
        updates = {"task": prepared.task.model_copy(update={"issue_body": "changed"})}
    else:
        updates = {
            "expectation": prepared.expectation.model_copy(
                update={"expected_exit_codes": (2,)}
            )
        }

    with pytest.raises(PostPatchReproductionError, match=f"{mismatch} fingerprint"):
        _verify(prepared, gateway, **updates)

    assert gateway.calls == []


def test_invalid_original_reproduction_is_rejected_before_execution(
    prepared: Prepared,
) -> None:
    gateway = Gateway(_execution(status=ReproductionStatus.NOT_REPRODUCED))
    state = prepared.reproduction_result.state.model_copy(update={"phase": AgentPhase.FAILED})
    invalid = prepared.reproduction_result.model_copy(update={"state": state})

    with pytest.raises(PostPatchReproductionError, match="original reproduction result"):
        _verify(prepared, gateway, original_reproduction_result=invalid)

    assert gateway.calls == []


def test_proposal_digest_mismatch_is_rejected_before_execution(prepared: Prepared) -> None:
    gateway = Gateway(_execution(status=ReproductionStatus.NOT_REPRODUCED))
    invalid = prepared.proposal.model_copy(update={"proposal_digest": "0" * 64})

    with pytest.raises(PostPatchReproductionError, match="proposal digest"):
        _verify(prepared, gateway, proposal=invalid)

    assert gateway.calls == []


def test_nonapplied_application_result_is_rejected_before_execution(
    prepared: Prepared,
) -> None:
    gateway = Gateway(_execution(status=ReproductionStatus.NOT_REPRODUCED))
    invalid = prepared.application_result.model_copy(update={"status": "not_applied"})

    with pytest.raises(PostPatchReproductionError, match="requires an applied"):
        _verify(prepared, gateway, application_result=invalid)

    assert gateway.calls == []


def test_application_candidate_metadata_mismatch_is_rejected_before_execution(
    prepared: Prepared,
) -> None:
    gateway = Gateway(_execution(status=ReproductionStatus.NOT_REPRODUCED))
    changed = prepared.application_result.files[0].model_copy(
        update={"candidate_file_sha256": "f" * 64}
    )
    invalid = prepared.application_result.model_copy(update={"files": (changed,)})

    with pytest.raises(PostPatchReproductionError, match="file metadata"):
        _verify(prepared, gateway, application_result=invalid)

    assert gateway.calls == []


def test_stale_workspace_candidate_is_rejected_before_execution(prepared: Prepared) -> None:
    gateway = Gateway(_execution(status=ReproductionStatus.NOT_REPRODUCED))
    prepared.source.write_bytes(b"stale\n")

    with pytest.raises(PostPatchReproductionError, match="applied candidate"):
        _verify(prepared, gateway)

    assert gateway.calls == []


@pytest.mark.parametrize("identity", ["command_id", "argv"])
def test_command_identity_mismatch_is_rejected_before_classification(
    prepared: Prepared, identity: str
) -> None:
    execution = _execution(status=ReproductionStatus.NOT_REPRODUCED)
    update = {"command_id": "other"} if identity == "command_id" else {"argv": ("other",)}
    gateway = Gateway(execution.model_copy(update=update))

    with pytest.raises(PostPatchReproductionError, match="command ID|arguments"):
        _verify(prepared, gateway)

    assert gateway.calls == ["unit_tests"]


def test_operational_execution_error_propagates_without_retry(prepared: Prepared) -> None:
    error = ApprovedCommandExecutionError("bounded execution failed")
    gateway = Gateway(error=error)

    with pytest.raises(ApprovedCommandExecutionError) as caught:
        _verify(prepared, gateway)

    assert caught.value is error
    assert gateway.calls == ["unit_tests"]


def test_command_induced_target_mutation_is_rejected(prepared: Prepared) -> None:
    gateway = Gateway(
        _execution(status=ReproductionStatus.NOT_REPRODUCED),
        mutation=(prepared.source, b"command mutation\n"),
    )

    with pytest.raises(PostPatchReproductionError, match="command modified"):
        _verify(prepared, gateway)

    assert gateway.calls == ["unit_tests"]
