"""Read-only agent state, actions, and investigation loop."""

from repofix.agent.actions import (
    AgentAction,
    FinishInvestigationAction,
    ListFilesAction,
    ReadFileAction,
    RecordHypothesisAction,
    SearchCodeAction,
    UnderstandIssueAction,
)
from repofix.agent.interfaces import AgentModel, ReadOnlyToolGateway, ToolExecutionError
from repofix.agent.loop import AgentProtocolError, run_read_only_investigation
from repofix.agent.state import (
    AgentPhase,
    AgentState,
    IssueUnderstanding,
    RepairHypothesis,
    ToolObservation,
)

__all__ = [
    "AgentAction",
    "AgentModel",
    "AgentPhase",
    "AgentProtocolError",
    "AgentState",
    "FinishInvestigationAction",
    "IssueUnderstanding",
    "ListFilesAction",
    "ReadFileAction",
    "ReadOnlyToolGateway",
    "RecordHypothesisAction",
    "RepairHypothesis",
    "SearchCodeAction",
    "ToolObservation",
    "ToolExecutionError",
    "UnderstandIssueAction",
    "run_read_only_investigation",
]
