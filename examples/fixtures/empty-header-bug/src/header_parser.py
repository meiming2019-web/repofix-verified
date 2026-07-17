"""Small header-value parser used by the read-only investigation example."""

DEFAULT_VALUE = "default"


def parse_header(header: str, configured_value: str) -> str:
    """Return the incoming header, falling back to the configured value when empty."""
    if header == "":
        return DEFAULT_VALUE
    return header
