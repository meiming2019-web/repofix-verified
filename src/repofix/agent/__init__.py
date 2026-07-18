"""Read-only agent state, actions, and investigation loop."""

from repofix.agent.actions import (
    AgentAction,
    FinishInvestigationAction,
    ListFilesAction,
    ReadFileAction,
    RecordHypothesisAction,
    RunApprovedCommandAction,
    SearchCodeAction,
    UnderstandIssueAction,
)
from repofix.agent.interfaces import AgentModel, ReadOnlyToolGateway, ToolExecutionError
from repofix.agent.loop import AgentProtocolError, run_read_only_investigation
from repofix.agent.reproduction_loop import (
    ApprovedCommandGateway,
    EvaluatorReproductionAttempt,
    ReproductionAgentRunResult,
    run_reproduction_agent_loop,
)
from repofix.agent.state import (
    AgentPhase,
    AgentReproductionObservation,
    AgentState,
    AgentWorkflow,
    IssueUnderstanding,
    RepairHypothesis,
    ToolObservation,
)

__all__ = [
    "AgentAction",
    "AgentModel",
    "AgentPhase",
    "AgentProtocolError",
    "AgentReproductionObservation",
    "AgentState",
    "AgentWorkflow",
    "ApprovedCommandGateway",
    "EvaluatorReproductionAttempt",
    "FinishInvestigationAction",
    "IssueUnderstanding",
    "ListFilesAction",
    "ReadFileAction",
    "ReadOnlyToolGateway",
    "RecordHypothesisAction",
    "ReproductionAgentRunResult",
    "RepairHypothesis",
    "SearchCodeAction",
    "RunApprovedCommandAction",
    "ToolObservation",
    "ToolExecutionError",
    "UnderstandIssueAction",
    "run_read_only_investigation",
    "run_reproduction_agent_loop",
]
