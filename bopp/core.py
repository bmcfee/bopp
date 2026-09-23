from typing import Any, TypeVar

import msgspec

from .registries import get_registry

__all__ = ["create"]

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
    bopp_version: str = "v1",
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
    bopp_version : str, default "v1"
        Schema version string to select the underlying type registry.
    **kwargs : Any
        Keyword arguments matching fields for the payload, extent, or
        confidence structures.

    Returns
    -------
    Annotation
        An instantiated Annotation structure.

    Raises
    ------
    ValueError
        If an unrecognized `payload_kind`, `extent_kind`, or `confidence_kind`
        is provided, or if unused keyword arguments remain.
    """
    registry = get_registry(bopp_version)
    
    PAYLOAD_TYPE_REGISTRY = registry["PAYLOAD_TYPE_REGISTRY"]
    EXTENT_TYPE_REGISTRY = registry["EXTENT_TYPE_REGISTRY"]
    CONFIDENCE_TYPE_REGISTRY = registry["CONFIDENCE_TYPE_REGISTRY"]
    Annotation = registry["Annotation"]

    try:
        payload_cls = PAYLOAD_TYPE_REGISTRY[payload_kind]
    except KeyError as e:
        raise ValueError(f"Unrecognized kind identifier: {e}") from e

    payload_args = _extract_kwargs(payload_cls, kwargs)
    payload_obj = payload_cls(**payload_args)

    # Extract kwargs by mutating the dictionary
    extent_obj = msgspec.UNSET
    if extent_kind is not None:
        try: 
            extent_cls = EXTENT_TYPE_REGISTRY[extent_kind]
        except KeyError as e:
            raise ValueError(f"Unrecognized extent kind {e}") from e

        extent_args = _extract_kwargs(extent_cls, kwargs)
        extent_obj = extent_cls(**extent_args)

    confidence_obj = msgspec.UNSET
    if confidence_kind:
        try:
            confidence_cls = CONFIDENCE_TYPE_REGISTRY[confidence_kind]
        except KeyError as e:
            raise ValueError(f"Unrecognized confidence kind: {e}") from e
        
        confidence_args = _extract_kwargs(confidence_cls, kwargs)
        confidence_obj = confidence_cls(**confidence_args)

    # Any remaining kwargs indicate a user typo or a schema mismatch
    if kwargs:
        raise ValueError(f"Unconsumed keyword arguments: {list(kwargs.keys())}")

    return Annotation(
        media_id=media_id,
        bopp_version=bopp_version,
        extent=extent_obj,
        payload=payload_obj,
        confidence=confidence_obj
    )



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
    ValueError
        If validation fails or if `target_type` is missing and cannot be inferred.
    """
    if target_type is None:
        if isinstance(obj, msgspec.Struct):
            target_type = type(obj)
        else:
            raise ValueError("target_type must be provided if obj is not a msgspec.Struct.")
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
        raise ValueError(f"Validation failed for {target_type.__name__}: {err}") from err
    return True
