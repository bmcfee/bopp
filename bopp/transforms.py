from __future__ import annotations

import copy
from typing import Any

import msgspec

from .base import BoppBase
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
    reset_time: bool = False,
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
    reset_time : bool, default False
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

    if reset_time and start is None:
        raise BoppArgumentError("reset_time=True requires 'start' to be specified.")

    extent = annotation.extent
    if extent is msgspec.UNSET or extent is None:
        # Return deep copy of annotation if no extent present
        return copy.deepcopy(annotation)

    extent_tag = _get_tag(extent)
    if extent_tag not in ("time", "time_interval", "time_frequency_box"):
        raise BoppArgumentError(
            f"Unsupported extent type for trim: '{extent_tag}'. "
            "Extent must be 'time', 'time_interval', or 'time_frequency_box'."
        )

    # Determine original observation count and primary time field
    kept_indices: list[int] = []
    updated_time_data: list[Any] = []

    shift = start if (reset_time and start is not None) else 0.0

    if extent_tag == "time":
        time_field = None
        for f in msgspec.structs.fields(type(extent)):
            val = getattr(extent, f.name)
            if isinstance(val, list):
                time_field = f.name
                break

        if time_field is None:
            raise BoppArgumentError("Extent 'time' structure contains no list field.")

        times: list[float] = getattr(extent, time_field)
        n_obs = len(times)

        for i, t in enumerate(times):
            if start is not None and t < start:
                continue
            if end is not None and t > end:
                continue
            kept_indices.append(i)
            updated_time_data.append(t - shift)

    elif extent_tag == "time_interval":
        interval_field = None
        for f in msgspec.structs.fields(type(extent)):
            val = getattr(extent, f.name)
            if isinstance(val, list):
                interval_field = f.name
                break

        if interval_field is None:
            raise BoppArgumentError("Extent 'time_interval' structure contains no list field.")

        intervals: list[list[float]] = getattr(extent, interval_field)
        n_obs = len(intervals)

        for i, interval in enumerate(intervals):
            t_min, t_max = interval[0], interval[1]
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
            updated_time_data.append([c_min - shift, c_max - shift])

    elif extent_tag == "time_frequency_box":
        box_field = None
        for f in msgspec.structs.fields(type(extent)):
            val = getattr(extent, f.name)
            if isinstance(val, list):
                box_field = f.name
                break

        if box_field is None:
            raise BoppArgumentError("Extent 'time_frequency_box' structure contains no list field.")

        boxes: list[list[float]] = getattr(extent, box_field)
        n_obs = len(boxes)

        for i, box in enumerate(boxes):
            t_min, t_max = box[0], box[1]
            f_min, f_max = box[2], box[3]
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
            updated_time_data.append([c_min - shift, c_max - shift, f_min, f_max])

    # Filter extent parallel fields
    extent_updates = {}
    for fname, fval in _get_list_fields_with_length(extent, n_obs).items():
        if fname in (time_field if extent_tag == "time" else (interval_field if extent_tag == "time_interval" else box_field),):
            extent_updates[fname] = updated_time_data
        else:
            extent_updates[fname] = [fval[idx] for idx in kept_indices]
    new_extent = msgspec.structs.replace(extent, **extent_updates)

    # Filter payload parallel fields
    payload = annotation.payload
    if payload is not msgspec.UNSET and payload is not None:
        payload_updates = {}
        for fname, fval in _get_list_fields_with_length(payload, n_obs).items():
            payload_updates[fname] = [fval[idx] for idx in kept_indices]
        new_payload = msgspec.structs.replace(payload, **payload_updates)
    else:
        new_payload = payload

    # Filter confidence parallel fields
    confidence = annotation.confidence
    if confidence is not msgspec.UNSET and confidence is not None:
        confidence_updates = {}
        for fname, fval in _get_list_fields_with_length(confidence, n_obs).items():
            confidence_updates[fname] = [fval[idx] for idx in kept_indices]
        new_confidence = msgspec.structs.replace(confidence, **confidence_updates)
    else:
        new_confidence = confidence

    new_sandbox = copy.deepcopy(annotation.sandbox) if annotation.sandbox is not msgspec.UNSET else msgspec.UNSET

    return msgspec.structs.replace(
        annotation,
        extent=new_extent,
        payload=new_payload,
        confidence=new_confidence,
        sandbox=new_sandbox,
    )
