"""Tests for the RepoFix command-line interface."""

from typer.testing import CliRunner

from repofix.cli import app

runner = CliRunner()


def test_version_command() -> None:
    """The version command should print the current project version."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "RepoFix Verified 0.1.0" in result.stdout