from __future__ import annotations

import uuid
from typing import Any, TypeVar, cast

import msgspec

from ._version import get_current_schema_version
from .exceptions import (
    BoppArgumentError,
    BoppArrayError,
    BoppError,
    BoppIOError,
    BoppRegistryError,
    BoppValidationError,
)
from .registries import get_registry

BOPP_NAMESPACE = uuid.UUID("a9b3e1c0-42f8-4e89-8d62-b9123456789a")

__all__ = [
    "BOPP_NAMESPACE",
    "BoppArgumentError",
    "BoppArrayError",
    "BoppError",
    "BoppIOError",
    "BoppRegistryError",
    "BoppValidationError",
    "compute_annotation_id",
    "create",
    "ensure_annotation_id",
    "validate",
    "validate_and_set_annotation_id",
    "validate_annotation_id",
]


def compute_annotation_id(annotation: Any) -> str:
    """
    Compute a deterministic UUIDv5 identifier for an annotation.

    The UUID is calculated over the canonical JSON serialization of the
    annotation content, excluding any existing 'id' field.

    Parameters
    ----------
    annotation : Any
        An Annotation struct instance or dictionary representation of an annotation.

    Returns
    -------
    str
        The computed UUIDv5 string representation.
    """

    data = msgspec.to_builtins(annotation, order="deterministic")
    if 'id' in data:
        del data['id']
    canonical_bytes = msgspec.json.encode(data, order="deterministic")
    return str(uuid.uuid5(BOPP_NAMESPACE, canonical_bytes))


def validate_annotation_id(annotation: Any) -> bool:
    """
    Validate the UUIDv5 identifier of an annotation without silently mutating it.

    If an 'id' is present, validates it against the computed UUIDv5.

    Parameters
    ----------
    annotation : Any
        An Annotation struct instance or dictionary representation of an annotation.

    Returns
    -------
    bool
        True if the ID is present and valid, False if ID is missing.

    Raises
    ------
    BoppValidationError
        If an existing 'id' on the annotation does not match the computed ID.
    """
    computed_id = compute_annotation_id(annotation)

    existing_id = (
        getattr(annotation, "id", None)
        if isinstance(annotation, msgspec.Struct)
        else annotation.get("id")
    )

    if existing_id is None or existing_id is msgspec.UNSET:
        return False

    if existing_id != computed_id:
        raise BoppValidationError(
            f"Annotation ID mismatch: object has '{existing_id}', computed '{computed_id}'"
        )

    return True


def ensure_annotation_id(annotation: Any) -> str:
    """
    Compute and set the UUIDv5 identifier for an annotation if missing,
    or validate it if present.

    Parameters
    ----------
    annotation : Any
        An Annotation struct instance or dictionary representation.

    Returns
    -------
    str
        The assigned or validated UUIDv5 identifier.

    Raises
    ------
    BoppValidationError
        If an existing 'id' does not match the computed ID.
    """
    computed_id = compute_annotation_id(annotation)

    existing_id = (
        getattr(annotation, "id", None)
        if isinstance(annotation, msgspec.Struct)
        else annotation.get("id")
    )

    if existing_id is None or existing_id is msgspec.UNSET:
        if isinstance(annotation, msgspec.Struct):
            annotation.id = computed_id  # type: ignore[attr-defined]
        else:
            annotation["id"] = computed_id
    elif existing_id != computed_id:
        raise BoppValidationError(
            f"Annotation ID mismatch: object has '{existing_id}', computed '{computed_id}'"
        )

    return computed_id


def validate_and_set_annotation_id(annotation: Any) -> str:
    """
    Deprecated alias for ensure_annotation_id.
    """
    return ensure_annotation_id(annotation)


def _extract_kwargs(cls: type[msgspec.Struct], kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and remove keys belonging to the target msgspec.Struct class.

    Parameters
    ----------
    cls : type of msgspec.Struct
        Target class whose valid field names will be extracted.
    kwargs : dict of str to Any
        Dictionary of keyword arguments to extract matching keys from.
        This dictionary is mutated in place.

    Returns
    -------
    dict of str to Any
        Dictionary of extracted keyword arguments relevant to `cls`.
    """
    valid_keys = {f.name for f in msgspec.structs.fields(cls)}
    return {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid_keys}


def create(
    *,
    media_id: str,
    payload_kind: str,
    extent_kind: str | None = None,
    confidence_kind: str | None = None,
    parents: list[str] | Any = msgspec.UNSET,
    bopp_version: str | None = None,
    sandbox: Any = msgspec.UNSET,
    generate_id: bool = True,
    **kwargs: Any
) -> Any:
    """
    Create a new Annotation instance for the specified BOPP version.

    Parameters
    ----------
    media_id : str
        Unique media identifier associated with the annotation.
    payload_kind : str
        Kind identifier registered for the payload struct.
    extent_kind : str or None, optional
        Kind identifier registered for the extent struct, if applicable.
    confidence_kind : str or None, optional
        Kind identifier registered for the confidence struct, if applicable.
    parents : list of str, optional
        List of parent annotation UUIDs from which this annotation was derived.
    bopp_version : str or None, optional
        Schema version string to select the underlying type registry.
        If None, defaults to the current default schema version.
    sandbox : Any, optional
        Unstructured storage area for arbitrary user-defined data.
    generate_id : bool, default True
        If True, computes and assigns deterministic UUIDv5 `id` to the annotation.
    **kwargs : Any
        Keyword arguments matching fields for the payload, extent, or
        confidence structures.

    Returns
    -------
    Annotation
        An instantiated Annotation structure.

    Raises
    ------
    BoppRegistryError
        If an unrecognized `payload_kind`, `extent_kind`, or `confidence_kind`
        is provided.
    BoppArgumentError
        If unused keyword arguments remain.
    """
    if bopp_version is None:
        bopp_version = get_current_schema_version()

    registry = get_registry(bopp_version)
    
    PAYLOAD_TYPE_REGISTRY = registry["PAYLOAD_TYPE_REGISTRY"]
    EXTENT_TYPE_REGISTRY = registry["EXTENT_TYPE_REGISTRY"]
    CONFIDENCE_TYPE_REGISTRY = registry["CONFIDENCE_TYPE_REGISTRY"]
    Annotation = registry["Annotation"]

    try:
        payload_cls = PAYLOAD_TYPE_REGISTRY[payload_kind]
    except KeyError as e:
        raise BoppRegistryError(f"Unrecognized kind identifier: {e}") from e

    payload_args = _extract_kwargs(payload_cls, kwargs)
    payload_obj = payload_cls(**payload_args)

    # Extract kwargs by mutating the dictionary
    extent_obj = msgspec.UNSET
    if extent_kind is not None:
        try: 
            extent_cls = EXTENT_TYPE_REGISTRY[extent_kind]
        except KeyError as e:
            raise BoppRegistryError(f"Unrecognized extent kind {e}") from e

        extent_args = _extract_kwargs(extent_cls, kwargs)
        extent_obj = extent_cls(**extent_args)

    confidence_obj = msgspec.UNSET
    if confidence_kind:
        try:
            confidence_cls = CONFIDENCE_TYPE_REGISTRY[confidence_kind]
        except KeyError as e:
            raise BoppRegistryError(f"Unrecognized confidence kind: {e}") from e
        
        confidence_args = _extract_kwargs(confidence_cls, kwargs)
        confidence_obj = confidence_cls(**confidence_args)

    # Any remaining kwargs indicate a user typo or a schema mismatch
    if kwargs:
        raise BoppArgumentError(f"Unconsumed keyword arguments: {list(kwargs.keys())}")

    ann = Annotation(
        media_id=media_id,
        parents=parents,
        bopp_version=bopp_version,
        extent=extent_obj,
        payload=payload_obj,
        confidence=confidence_obj,
        sandbox=sandbox,
    )

    if generate_id:
        ensure_annotation_id(ann)
    else:
        validate_annotation_id(ann)

    return ann


T = TypeVar("T")


def validate(obj: Any, target_type: type[T] | None = None) -> bool:
    """
    Validate an object against a target msgspec structure type.

    Parameters
    ----------
    obj : Any
        Object to validate (e.g., a msgspec Struct instance or a dict).
    target_type : type of T or None, optional
        Target struct type to validate against. If None and `obj` is a
        `msgspec.Struct`, `type(obj)` is used.

    Returns
    -------
    bool
        True if validation succeeds.

    Raises
    ------
    BoppArgumentError
        If `target_type` is missing and cannot be inferred.
    BoppValidationError
        If validation fails.
    """
    if target_type is None:
        if isinstance(obj, msgspec.Struct):
            target_type = cast(type[T], type(obj))
        else:
            raise BoppArgumentError("target_type must be provided if obj is not a msgspec.Struct.")
    assert target_type is not None

    # 1. Handle Struct instances: convert to builtins to strip UNSET keys recursively
    if isinstance(obj, msgspec.Struct):
        obj = msgspec.to_builtins(obj)
    # 2. Handle raw dicts: filter out top-level UNSET sentinels if manually populated
    elif isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if v is not msgspec.UNSET}

    try:
        msgspec.convert(obj, target_type)
    except (msgspec.ValidationError, TypeError) as err:
        raise BoppValidationError(f"Validation failed for {target_type.__name__}: {err}") from err
    return True
