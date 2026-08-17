"""Narrow evaluator-only execution interface."""

from typing import Protocol

from repofix.execution import ApprovedCommandExecutionResult


class EvaluatorCommandGateway(Protocol):
    def execute(self, command_id: str) -> ApprovedCommandExecutionResult:
        """Execute one trusted evaluator-owned command by exact ID."""
        ...
