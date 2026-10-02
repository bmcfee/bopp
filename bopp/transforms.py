from __future__ import annotations

import copy
from typing import Any

import msgspec

from .base import BoppBase
from .core import validate_and_set_annotation_id
from .exceptions import BoppArgumentError
from .util import _get_tag


def _get_list_fields_with_length(struct: msgspec.Struct, expected_length: int) -> dict[str, list[Any]]:
    """Return dictionary of field name -> list value for list fields matching expected_length."""
    result = {}
    for field in msgspec.structs.fields(type(struct)):
        val = getattr(struct, field.name)
        if isinstance(val, list) and len(val) == expected_length:
            result[field.name] = val
    return result


def trim(
    annotation: BoppBase,
    *,
    start: float | None = None,
    end: float | None = None,
    strict: bool = False,
    reset: bool = False,
) -> BoppBase:
    """
    Trim an Annotation to a specific time range [start, end].

    Parameters
    ----------
    annotation : BoppBase
        The input Annotation model instance to trim.
    start : float or None, optional
        Start time in seconds. If None, no lower bound trimming is applied.
    end : float or None, optional
        End time in seconds. If None, no upper bound trimming is applied.
    strict : bool, default False
        If True, only observations entirely contained within [start, end] persist.
        If False, intervals/boxes overlapping boundary are clipped to bounds.
    reset : bool, default False
        If True, adjusts all time fields relative to `start` (t -> t - start).
        Requires `start` to be non-None when True.

    Returns
    -------
    BoppBase
        A new Annotation instance with trimmed extents, payloads, and confidences.

    Raises
    ------
    BoppArgumentError
        If arguments are invalid or the extent type is unsupported.
    """
    if start is None and end is None:
        raise BoppArgumentError("At least one of 'start' or 'end' must be provided.")

    if start is not None and end is not None and start > end:
        raise BoppArgumentError(f"start ({start}) must be <= end ({end}).")

    if reset and start is None:
        raise BoppArgumentError("reset=True requires 'start' to be specified.")

    # Transitively accumulate parent IDs
    existing_parents = getattr(annotation, "parents", None)
    if existing_parents is None or existing_parents is msgspec.UNSET:
        new_parents = [annotation.id]
    else:
        new_parents = list(existing_parents) + [annotation.id]

    extent = getattr(annotation, "extent", msgspec.UNSET)
    if extent is msgspec.UNSET or extent is None:
        # Return copy of annotation with updated parent lineage
        new_ann = msgspec.structs.replace(
            annotation,
            id=msgspec.UNSET,
            parents=new_parents,
            sandbox=copy.deepcopy(annotation.sandbox) if getattr(annotation, "sandbox", msgspec.UNSET) is not msgspec.UNSET else msgspec.UNSET,
        )
        validate_and_set_annotation_id(new_ann)
        return new_ann

    extent_tag = _get_tag(extent)
    if extent_tag not in ("time", "time_interval", "time_frequency_box"):
        raise BoppArgumentError(
            f"Unsupported extent type for trim: '{extent_tag}'. "
            "Extent must be 'time', 'time_interval', or 'time_frequency_box'."
        )

    kept_indices: list[int] = []
    shift = start if (reset and start is not None) else 0.0

    extent_updates: dict[str, list[Any]] = {}
    new_time: list[float] = []

    if extent_tag == "time":
        time_vals = extent.time  # type: ignore[union-attr]
        n_obs = len(time_vals)

        for i, t in enumerate(time_vals):
            if start is not None and t < start:
                continue
            if end is not None and t > end:
                continue
            kept_indices.append(i)
            new_time.append(t - shift)

        extent_updates["time"] = new_time

    elif extent_tag in ("time_interval", "time_frequency_box"):
        time_vals = extent.time  # type: ignore[union-attr]
        duration_vals = extent.duration  # type: ignore[union-attr]
        n_obs = len(time_vals)

        new_duration: list[float] = []

        for i, (t_min, dur) in enumerate(zip(time_vals, duration_vals)):
            t_max = t_min + dur

            if strict:
                if start is not None and t_min < start:
                    continue
                if end is not None and t_max > end:
                    continue
                c_min, c_max = t_min, t_max
            else:
                if start is not None and t_max <= start:
                    continue
                if end is not None and t_min >= end:
                    continue
                c_min = max(t_min, start) if start is not None else t_min
                c_max = min(t_max, end) if end is not None else t_max

            kept_indices.append(i)
            new_time.append(c_min - shift)
            new_duration.append(c_max - c_min)

        extent_updates["time"] = new_time
        extent_updates["duration"] = new_duration

    # Filter remaining parallel list fields in extent
    for fname, fval in _get_list_fields_with_length(extent, n_obs).items():
        if fname not in extent_updates:
            extent_updates[fname] = [fval[idx] for idx in kept_indices]

    new_extent = msgspec.structs.replace(extent, **extent_updates)

    # Filter payload parallel fields
    payload = getattr(annotation, "payload", msgspec.UNSET)
    payload_updates = {}
    for fname, fval in _get_list_fields_with_length(payload, n_obs).items():
        payload_updates[fname] = [fval[idx] for idx in kept_indices]
    new_payload = msgspec.structs.replace(payload, **payload_updates)

    # Filter confidence parallel fields
    confidence = getattr(annotation, "confidence", msgspec.UNSET)
    kwargs = {}
    if confidence is not msgspec.UNSET and confidence is not None:
        confidence_updates = {}
        for fname, fval in _get_list_fields_with_length(confidence, n_obs).items():
            confidence_updates[fname] = [fval[idx] for idx in kept_indices]
            kwargs["confidence"] = msgspec.structs.replace(confidence, **confidence_updates)

    sandbox = getattr(annotation, "sandbox", msgspec.UNSET)
    new_sandbox = copy.deepcopy(sandbox) if sandbox is not msgspec.UNSET else msgspec.UNSET

    new_ann = msgspec.structs.replace(
        annotation,
        id=msgspec.UNSET,
        parents=new_parents,
        extent=new_extent,
        payload=new_payload,
        sandbox=new_sandbox,
        **kwargs
    )
    validate_and_set_annotation_id(new_ann)
    return new_ann
