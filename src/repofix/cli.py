"""Command-line entry point for RepoFix Verified."""

import typer

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


if __name__ == "__main__":
    app()