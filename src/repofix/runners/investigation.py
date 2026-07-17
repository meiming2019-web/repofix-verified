"""Provider-independent orchestration for read-only investigations."""

import json
from pathlib import Path

from repofix.agent import AgentModel, AgentState, run_read_only_investigation
from repofix.tasks import load_agent_task_spec
from repofix.tools import LocalReadOnlyToolGateway


MAX_INVESTIGATION_STEPS = 20
MAX_REPORT_OBSERVATION_CHARS = 4_000
_TRUNCATION_MARKER = "...[observation output truncated]"


def run_investigation_from_paths(
    *,
    task_path: Path,
    workspace_root: Path,
    model: AgentModel,
    max_steps: int,
) -> AgentState:
    """Load a task and run the existing loop against a prepared local workspace."""
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or not 1 <= max_steps <= MAX_INVESTIGATION_STEPS
    ):
        raise ValueError("max_steps must be a strict integer from 1 through 20")

    task = load_agent_task_spec(task_path)
    tools = LocalReadOnlyToolGateway(
        workspace_root=workspace_root,
        allowed_source_paths=task.allowed_source_paths,
    )
    return run_read_only_investigation(
        task=task,
        model=model,
        tools=tools,
        max_steps=max_steps,
    )


def _json(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _escape_control_characters(serialized)


def _escape_control_characters(value: str, *, preserve_newlines: bool = False) -> str:
    escaped: list[str] = []
    for character in value:
        codepoint = ord(character)
        if character == "\n" and preserve_newlines:
            escaped.append(character)
        elif codepoint <= 0x1F or 0x7F <= codepoint <= 0x9F:
            escaped.append(f"\\u{codepoint:04x}")
        else:
            escaped.append(character)
    return "".join(escaped)


def _inline(value: str) -> str:
    return _escape_control_characters(value)


def _display_observation_output(output: str) -> str:
    sanitized = _escape_control_characters(output, preserve_newlines=True)
    truncated = len(sanitized) > MAX_REPORT_OBSERVATION_CHARS
    displayed = sanitized[:MAX_REPORT_OBSERVATION_CHARS]
    lines = [f"     | {line}" for line in displayed.split("\n")]
    if truncated:
        lines.append(f"     | {_TRUNCATION_MARKER}")
    return "\n".join(lines)


def render_investigation_report(state: AgentState) -> str:
    """Render deterministic, bounded plain text from public investigation state."""
    lines = [
        "RepoFix Read-Only Investigation",
        f"Task ID: {_inline(state.task_id)}",
        f"Phase: {state.phase.value}",
        f"Steps: {state.step_count}",
    ]

    if state.issue_understanding is not None:
        understanding = state.issue_understanding
        lines.extend(
            [
                "Issue understanding:",
                f"  Expected behavior: {_inline(understanding.expected_behavior)}",
                f"  Observed behavior: {_inline(understanding.observed_behavior)}",
                f"  Reproduction clues: {_json(list(understanding.reproduction_clues))}",
                f"  Likely components: {_json(list(understanding.likely_components))}",
                f"  Missing information: {_json(list(understanding.missing_information))}",
            ]
        )

    lines.append("Observations:")
    if not state.observations:
        lines.append("  None")
    for index, observation in enumerate(state.observations, start=1):
        lines.extend(
            [
                f"  {index}. Tool: {_inline(observation.tool_name)}",
                f"     Success: {_json(observation.success)}",
                f"     Arguments: {_json(observation.arguments)}",
            ]
        )
        if observation.success:
            lines.extend(
                [
                    "     Output:",
                    _display_observation_output(observation.output),
                ]
            )
        else:
            error = observation.error if observation.error is not None else ""
            lines.append(f"     Error: {_inline(error)}")

    lines.append("Hypotheses:")
    if not state.hypotheses:
        lines.append("  None")
    for index, hypothesis in enumerate(state.hypotheses, start=1):
        lines.extend(
            [
                f"  {index}. ID: {_inline(hypothesis.hypothesis_id)}",
                f"     Description: {_inline(hypothesis.description)}",
                f"     Confidence: {_json(hypothesis.confidence)}",
                f"     Status: {_inline(hypothesis.status)}",
                f"     Supporting evidence: {_json(list(hypothesis.supporting_evidence))}",
                f"     Contradicting evidence: {_json(list(hypothesis.contradicting_evidence))}",
            ]
        )

    if state.terminal_summary is not None:
        lines.append(f"Final summary: {_inline(state.terminal_summary)}")
    if state.failure_reason is not None:
        lines.append(f"Failure reason: {_inline(state.failure_reason)}")
    return "\n".join(lines)
