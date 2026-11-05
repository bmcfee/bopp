import importlib
from typing import Any

import msgspec

from .._version import get_registry_version
from ..exceptions import BoppRegistryError

FACET_REGISTRY_KEYS = {
    "extent": "EXTENT_TYPE_REGISTRY",
    "payload": "PAYLOAD_TYPE_REGISTRY",
    "confidence": "CONFIDENCE_TYPE_REGISTRY",
}


def get_registry(version: str | None = None) -> dict[str, Any]:
    """
    Retrieve the type registry mapping for the specified BOPP schema version.

    Parameters
    ----------
    version : str or None, optional
        Semantic version string of the BOPP schema (e.g., "1.0.0" or "v1").
        If None, defaults to the default schema version.

    Returns
    -------
    dict of str to Any
        Dictionary mapping registry keys to their corresponding types and mappings:
        - "PAYLOAD_TYPE_REGISTRY": dict mapping payload tags to payload Struct types.
        - "EXTENT_TYPE_REGISTRY": dict mapping extent tags to extent Struct types.
        - "CONFIDENCE_TYPE_REGISTRY": dict mapping confidence tags to confidence Struct types.
        - "METADATA_TYPE_REGISTRY": dict mapping metadata tags to metadata Struct types.
        - "COMPLEX_FIELDS_REGISTRY": dict mapping facet types to fields containing complex structures.
        - "STRING_FIELDS_REGISTRY": dict mapping facet types to fields containing string values.
        - "Annotation": the root Annotation Struct class for the version.
        - "Grouping": the Grouping Struct class for the version.

    Raises
    ------
    BoppRegistryError
        If the schema version does not match any known registry pattern.
    ValueError
        If the resolved registry version is unrecognized.

    Examples
    --------
    >>> from bopp.registries import get_registry
    >>> reg = get_registry()
    >>> "Annotation" in reg
    True
    """
    registry_key = get_registry_version(version)
    try:
        mod = importlib.import_module(f".{registry_key}", package=__name__)
    except ModuleNotFoundError:
        raise ValueError(
            f"Unknown registry version '{registry_key}' for schema '{version}'."
        ) from None

    return {
        "PAYLOAD_TYPE_REGISTRY": getattr(mod, "PAYLOAD_TYPE_REGISTRY", {}),
        "EXTENT_TYPE_REGISTRY": getattr(mod, "EXTENT_TYPE_REGISTRY", {}),
        "CONFIDENCE_TYPE_REGISTRY": getattr(mod, "CONFIDENCE_TYPE_REGISTRY", {}),
        "METADATA_TYPE_REGISTRY": getattr(mod, "METADATA_TYPE_REGISTRY", {}),
        "COMPLEX_FIELDS_REGISTRY": getattr(mod, "COMPLEX_FIELDS_REGISTRY", {}),
        "STRING_FIELDS_REGISTRY": getattr(mod, "STRING_FIELDS_REGISTRY", {}),
        "Annotation": getattr(mod, "ANNOTATION_CLASS", None),
        "Grouping": getattr(mod, "GROUPING_CLASS", None),
    }


def get_metadata_class(
    metadata_type: str, version: str | None = None
) -> type[msgspec.Struct]:
    """
    Retrieve the metadata Struct class for a given metadata type tag.

    Parameters
    ----------
    metadata_type : str
        The metadata type discriminator tag (e.g. 'human', 'algorithm', 'derived').
    version : str or None, optional
        Schema version string. If None, defaults to the default schema version.

    Returns
    -------
    type of msgspec.Struct
        The matching metadata Struct class.

    Raises
    ------
    BoppRegistryError
        If the metadata type is not registered.
    """
    reg = get_registry(version)["METADATA_TYPE_REGISTRY"]
    try:
        return reg[metadata_type]
    except KeyError:
        raise BoppRegistryError(
            f"Unrecognized metadata type: '{metadata_type}'. Available: {sorted(reg.keys())}"
        ) from None


def get_grouping_class(version: str | None = None) -> type[msgspec.Struct]:
    """
    Retrieve the Grouping Struct class for a given schema version.

    Parameters
    ----------
    version : str or None, optional
        Schema version string. If None, defaults to the default schema version.

    Returns
    -------
    type of msgspec.Struct
        The Grouping Struct class.
    """
    grouping_cls = get_registry(version).get("Grouping")
    if grouping_cls is None:
        raise BoppRegistryError(
            f"No Grouping class registered for version '{version}'."
        )
    return grouping_cls


def get_payload_class(
    payload_type: str, version: str | None = None
) -> type[msgspec.Struct]:
    """
    Retrieve the payload Struct class for a given payload type tag.

    Parameters
    ----------
    payload_type : str
        The payload type discriminator tag (e.g. 'tag', 'chord', 'beat').
    version : str or None, optional
        Schema version string. If None, defaults to the default schema version.

    Returns
    -------
    type of msgspec.Struct
        The matching payload Struct class.

    Raises
    ------
    BoppRegistryError
        If the payload type is not registered.
    """
    reg = get_registry(version)["PAYLOAD_TYPE_REGISTRY"]
    try:
        return reg[payload_type]
    except KeyError:
        raise BoppRegistryError(
            f"Unrecognized payload type: '{payload_type}'. Available: {sorted(reg.keys())}"
        ) from None


def get_extent_class(
    extent_type: str, version: str | None = None
) -> type[msgspec.Struct]:
    """
    Retrieve the extent Struct class for a given extent type tag.

    Parameters
    ----------
    extent_type : str
        The extent type discriminator tag (e.g. 'time', 'time_interval', 'pixel').
    version : str or None, optional
        Schema version string. If None, defaults to the default schema version.

    Returns
    -------
    type of msgspec.Struct
        The matching extent Struct class.

    Raises
    ------
    BoppRegistryError
        If the extent type is not registered.
    """
    reg = get_registry(version)["EXTENT_TYPE_REGISTRY"]
    try:
        return reg[extent_type]
    except KeyError:
        raise BoppRegistryError(
            f"Unrecognized extent type: '{extent_type}'. Available: {sorted(reg.keys())}"
        ) from None


def get_confidence_class(
    confidence_type: str, version: str | None = None
) -> type[msgspec.Struct]:
    """
    Retrieve the confidence Struct class for a given confidence type tag.

    Parameters
    ----------
    confidence_type : str
        The confidence type discriminator tag (e.g. 'likelihood', 'variance', 'agreement').
    version : str or None, optional
        Schema version string. If None, defaults to the default schema version.

    Returns
    -------
    type of msgspec.Struct
        The matching confidence Struct class.

    Raises
    ------
    BoppRegistryError
        If the confidence type is not registered.
    """
    reg = get_registry(version)["CONFIDENCE_TYPE_REGISTRY"]
    try:
        return reg[confidence_type]
    except KeyError:
        raise BoppRegistryError(
            f"Unrecognized confidence type: '{confidence_type}'. Available: {sorted(reg.keys())}"
        ) from None


def get_annotation_class(version: str | None = None) -> type[msgspec.Struct]:
    """
    Retrieve the root Annotation Struct class for a given schema version.

    Parameters
    ----------
    version : str or None, optional
        Schema version string. If None, defaults to the default schema version.

    Returns
    -------
    type of msgspec.Struct
        The root Annotation Struct class.
    """
    ann_cls = get_registry(version).get("Annotation")
    if ann_cls is None:
        raise BoppRegistryError(
            f"No Annotation class registered for version '{version}'."
        )
    return ann_cls
