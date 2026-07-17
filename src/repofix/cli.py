"""Command-line entry point for RepoFix Verified."""

from pathlib import Path
from typing import Annotated, Never

import typer
from openai import OpenAIError

from repofix.agent import AgentPhase, AgentProtocolError, ToolExecutionError
from repofix.agent.prompts import PromptConstructionError
from repofix.models import ModelExecutionError, OpenAIResponsesAgentModel
from repofix.runners import render_investigation_report, run_investigation_from_paths
from repofix.tasks import TaskSpecLoadError

app = typer.Typer(
    name="repofix",
    help="Reproduce-first, independently verified AI code repair.",
)


@app.callback()
def main() -> None:
    """RepoFix Verified command-line interface."""


@app.command()
def version() -> None:
    """Print the current project version."""
    typer.echo("RepoFix Verified 0.1.0")


def _operational_failure(message: str) -> Never:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


@app.command()
def investigate(
    task: Annotated[
        Path,
        typer.Option(
            "--task",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Agent-visible YAML task specification.",
        ),
    ],
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Prepared local repository workspace.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="OpenAI model name for investigation decisions."),
    ],
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=1, max=20, help="Maximum model decisions."),
    ] = 8,
) -> None:
    """Run a real-model, read-only investigation of a prepared workspace."""
    if not model.strip():
        _operational_failure("model name must be nonempty")

    try:
        agent_model = OpenAIResponsesAgentModel(model=model)
        state = run_investigation_from_paths(
            task_path=task,
            workspace_root=workspace,
            model=agent_model,
            max_steps=max_steps,
        )
    except OpenAIError:
        _operational_failure("OpenAI client setup failed")
    except (
        TaskSpecLoadError,
        AgentProtocolError,
        PromptConstructionError,
        ModelExecutionError,
        ToolExecutionError,
    ) as error:
        _operational_failure(str(error))

    typer.echo(render_investigation_report(state))
    if state.phase is AgentPhase.FAILED:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
