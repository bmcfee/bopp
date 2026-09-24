"""Version information for bopp."""

from typing import Final

__version__: Final[str]

__version__ = "0.0.1dev"

DEFAULT_SCHEMA_VERSION: Final[str] = "1.0"

SCHEMA_TO_REGISTRY: Final[dict[str, str]] = {
    "1.0": "v1",
    "1": "v1",
    "v1": "v1",
}


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
    ValueError
        If the schema version is not supported in SCHEMA_TO_REGISTRY.
    """
    version = schema_version or DEFAULT_SCHEMA_VERSION
    try:
        return SCHEMA_TO_REGISTRY[version]
    except KeyError as err:
        raise ValueError(
            f"Unsupported schema version: {version!r}. "
            f"Supported versions: {list(SCHEMA_TO_REGISTRY.keys())}"
        ) from err
