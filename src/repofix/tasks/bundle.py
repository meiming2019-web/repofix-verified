"""One authoritative evaluator task bundle for verified workflows."""

from typing import Self

from pydantic import model_validator

from repofix.reproduction.models import ReproductionExpectation
from repofix.tasks.spec import (
    AgentTaskSpec,
    GoldPatchSpec,
    HiddenVerificationSpecification,
    PatchPolicySpecification,
    RegressionSpecification,
    StrictFrozenModel,
)


class EvaluatorTaskBundle(StrictFrozenModel):
    """Complete evaluator data with an explicit agent-facing boundary."""

    task: AgentTaskSpec
    reproduction: ReproductionExpectation
    regression: RegressionSpecification
    hidden_verification: HiddenVerificationSpecification
    patch_policy: PatchPolicySpecification
    gold_patch: GoldPatchSpec

    @model_validator(mode="after")
    def validate_command_references(self) -> Self:
        if self.reproduction.command_id not in self.task.approved_commands:
            raise ValueError("reproduction command ID is not an approved task command")
        if self.regression.command_id not in self.task.approved_commands:
            raise ValueError("regression command ID is not an approved task command")
        if self.hidden_verification.command_id in self.task.approved_commands:
            raise ValueError("hidden command ID must not be agent-visible")
        return self

    def agent_view(self) -> AgentTaskSpec:
        """Return only the task information intended for the agent."""
        return self.task
