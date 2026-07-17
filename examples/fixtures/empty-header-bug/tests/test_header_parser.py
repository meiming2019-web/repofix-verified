"""Expected behavior for the controlled header-parser fixture."""

from header_parser import parse_header


def test_empty_header_retains_configured_value() -> None:
    assert parse_header("", "configured") == "configured"


def test_nonempty_header_is_returned() -> None:
    assert parse_header("incoming", "configured") == "incoming"
