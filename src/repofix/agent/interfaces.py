"""Typed boundaries for models and read-only tools."""

from dataclasses import dataclass
from typing import Protocol

from repofix.agent.actions import AgentAction
from repofix.agent.state import AgentState
from repofix.tasks import AgentTaskSpec


class ToolExecutionError(RuntimeError):
    """Raised when a tool operation fails for an expected runtime reason."""


@dataclass(frozen=True)
class ReadFileResult:
    """Rendered source range plus trusted full-file integrity metadata."""

    output: str
    full_file_sha256: str


class AgentModel(Protocol):
    """A model that selects the next structured investigation action."""

    def next_action(self, *, task: AgentTaskSpec, state: AgentState) -> AgentAction:
        """Select the next action from the current public state."""
        ...


class ReadOnlyToolGateway(Protocol):
    """Read-only repository exploration operations."""

    def list_files(self, path: str) -> str:
        """Return a textual directory listing."""
        ...

    def search_code(self, query: str, file_glob: str | None = None) -> str:
        """Return textual code-search matches."""
        ...

    def read_file(self, path: str, start_line: int, end_line: int) -> str:
        """Return a textual source range."""
        ...

    def read_file_with_metadata(self, path: str, start_line: int, end_line: int) -> ReadFileResult:
        """Return a source range and a system-computed full-file hash."""
        ...
