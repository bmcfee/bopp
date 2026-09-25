"""Version information for bopp."""

import re
from typing import Final

from .exceptions import BoppRegistryError

__version__ = "0.0.1dev"

DEFAULT_SCHEMA_VERSION: Final[str] = "1.0"

SCHEMA_PATTERNS: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(r"^1(?:\.\d+)*$|^v1$"), "v1"),
]


def get_current_schema_version() -> str:
    """Return the default schema version for BOPP models.

    Returns
    -------
    str
        The default schema version string.
    """
    return DEFAULT_SCHEMA_VERSION


def get_registry_version(schema_version: str | None = None) -> str:
    """Map a schema version string to its registry key (e.g. '1.0' -> 'v1').

    Parameters
    ----------
    schema_version : str or None, optional
        The schema version string. If None, uses DEFAULT_SCHEMA_VERSION.

    Returns
    -------
    str
        The registry module key (e.g., 'v1').

    Raises
    ------
    BoppRegistryError
        If the schema version does not match any known registry pattern.
    """
    version = schema_version or DEFAULT_SCHEMA_VERSION
    for pattern, registry_key in SCHEMA_PATTERNS:
        if pattern.match(version):
            return registry_key

    raise BoppRegistryError(f"Unsupported schema version: {version!r}.")
