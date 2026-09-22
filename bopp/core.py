from typing import Any

import msgspec

from .registries import get_registry

__all__ = ["create"]

def _extract_kwargs(cls: type[msgspec.Struct], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract and remove keys belonging to the target msgspec.Struct."""
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
