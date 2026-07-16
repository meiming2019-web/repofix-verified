"""Task specification models and loading APIs."""

from repofix.tasks.loader import TaskSpecLoadError, load_agent_task_spec, load_evaluator_task_bundle
from repofix.tasks.spec import (
    AgentTaskSpec,
    ApprovedCommand,
    EvaluatorTaskBundle,
    GoldPatchSpec,
    HiddenTestSpec,
)

__all__ = [
    "AgentTaskSpec",
    "ApprovedCommand",
    "EvaluatorTaskBundle",
    "GoldPatchSpec",
    "HiddenTestSpec",
    "TaskSpecLoadError",
    "load_agent_task_spec",
    "load_evaluator_task_bundle",
]
