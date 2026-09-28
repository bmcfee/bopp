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

    kept_indices: list[int] = []
    shift = start if (reset_time and start is not None) else 0.0

    extent_updates: dict[str, list[Any]] = {}

    if extent_tag == "time":
        if not hasattr(extent, "times") or getattr(extent, "times") is None:
            raise BoppArgumentError("Extent 'time' structure missing 'times' field.")

        times: list[float] = getattr(extent, "times")
        n_obs = len(times)
        new_times: list[float] = []

        for i, t in enumerate(times):
            if start is not None and t < start:
                continue
            if end is not None and t > end:
                continue
            kept_indices.append(i)
            new_times.append(t - shift)

        extent_updates["times"] = new_times

    elif extent_tag in ("time_interval", "time_frequency_box"):
        if not hasattr(extent, "times") or not hasattr(extent, "durations"):
            raise BoppArgumentError(f"Extent '{extent_tag}' structure missing 'times' or 'durations' field.")

        times: list[float] = getattr(extent, "times")
        durations: list[float] = getattr(extent, "durations")
        n_obs = len(times)

        new_times: list[float] = []
        new_durations: list[float] = []

        for i, (t_min, dur) in enumerate(zip(times, durations)):
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
            new_times.append(c_min - shift)
            new_durations.append(c_max - c_min)

        extent_updates["times"] = new_times
        extent_updates["durations"] = new_durations

    # Filter remaining parallel list fields in extent
    for fname, fval in _get_list_fields_with_length(extent, n_obs).items():
        if fname not in extent_updates:
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
