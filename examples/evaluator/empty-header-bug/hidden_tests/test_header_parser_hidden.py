"""Evaluator-only behavior not exposed through the agent workspace."""

from header_parser import parse_header


def test_empty_header_retains_alternate_configured_value() -> None:
    assert parse_header("", "alternate-configured") == "alternate-configured"
