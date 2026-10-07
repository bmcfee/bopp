from typing import Any

from .._version import get_registry_version


def get_registry(version: str) -> dict[str, Any]:
    """
    Retrieve the type registry mapping for the specified BOPP schema version.

    Parameters
    ----------
    version : str
        Semantic version string of the BOPP schema (e.g., "1.0.0" or "v1").

    Returns
    -------
    dict of str to Any
        Dictionary mapping registry keys to their corresponding types and mappings:
        - "PAYLOAD_TYPE_REGISTRY": dict mapping payload tags to payload Struct types.
        - "EXTENT_TYPE_REGISTRY": dict mapping extent tags to extent Struct types.
        - "CONFIDENCE_TYPE_REGISTRY": dict mapping confidence tags to confidence Struct types.
        - "COMPLEX_FIELDS_REGISTRY": dict mapping facet types to fields containing complex structures.
        - "Annotation": the root Annotation Struct class for the version.

    Raises
    ------
    ValueError
        If the resolved registry version is unrecognized.

    Examples
    --------
    >>> from bopp.registries import get_registry
    >>> reg = get_registry("1.0.0")
    >>> "Annotation" in reg
    True
    """
    registry_key = get_registry_version(version)

    if registry_key == "v1":
        from . import v1

        return {
            "PAYLOAD_TYPE_REGISTRY": v1.PAYLOAD_TYPE_REGISTRY,
            "EXTENT_TYPE_REGISTRY": v1.EXTENT_TYPE_REGISTRY,
            "CONFIDENCE_TYPE_REGISTRY": v1.CONFIDENCE_TYPE_REGISTRY,
            "COMPLEX_FIELDS_REGISTRY": getattr(v1, "COMPLEX_FIELDS_REGISTRY", {}),
            "Annotation": v1.ANNOTATION_CLASS,
        }

    raise ValueError(f"Unknown registry version '{registry_key}' for schema '{version}'.")
