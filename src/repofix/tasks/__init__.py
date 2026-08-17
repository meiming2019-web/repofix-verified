"""Task specification models and loading APIs."""

from importlib import import_module
from typing import TYPE_CHECKING

from repofix.tasks.spec import (
    AgentTaskSpec,
    ApprovedCommand,
    EvaluatorFileReference,
    GoldPatchSpec,
    HiddenVerificationSpecification,
    RegressionSpecification,
)


_LAZY_EXPORTS = {
    "EvaluatorTaskBundle": ("repofix.tasks.bundle", "EvaluatorTaskBundle"),
    "TaskSpecLoadError": ("repofix.tasks.loader", "TaskSpecLoadError"),
    "load_agent_task_spec": ("repofix.tasks.loader", "load_agent_task_spec"),
    "load_evaluator_task_bundle": (
        "repofix.tasks.loader",
        "load_evaluator_task_bundle",
    ),
    "load_reproduction_task_bundle": (
        "repofix.tasks.loader",
        "load_reproduction_task_bundle",
    ),
}

if TYPE_CHECKING:
    from repofix.tasks.bundle import EvaluatorTaskBundle
    from repofix.tasks.loader import (
        TaskSpecLoadError,
        load_agent_task_spec,
        load_evaluator_task_bundle,
        load_reproduction_task_bundle,
    )


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AgentTaskSpec",
    "ApprovedCommand",
    "EvaluatorFileReference",
    "EvaluatorTaskBundle",
    "GoldPatchSpec",
    "HiddenVerificationSpecification",
    "RegressionSpecification",
    "TaskSpecLoadError",
    "load_agent_task_spec",
    "load_evaluator_task_bundle",
    "load_reproduction_task_bundle",
]
