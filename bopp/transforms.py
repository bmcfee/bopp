from __future__ import annotations

import copy
import warnings
from typing import Any

import msgspec

from .base import BoppBase
from .core import validate_and_set_annotation_id
from .exceptions import BoppArgumentError
from .util import _get_tag

# Mapping default target_field when target_field is None
DEFAULT_TARGET_FIELDS: dict[str, str] = {
    "time": "time",
    "time_interval": "time",
    "time_frequency_box": "time",
    "midi_tick": "tick",
    "midi_interval": "tick",
    "score_quarter": "quarter",
    "score_interval": "quarter",
}

# Axis structure definition: (extent_tag, target_field) -> (kind, origin_or_min, span_or_max)
# kind can be "point", "origin_span", or "min_max"
AXIS_CONFIGS: dict[tuple[str, str], tuple[str, str, str | None]] = {
    ("time", "time"): ("point", "time", None),
    ("time_interval", "time"): ("origin_span", "time", "duration"),
    ("time_frequency_box", "time"): ("origin_span", "time", "duration"),
    ("time_frequency_box", "frequency"): ("min_max", "freq_min", "freq_max"),
    ("pixel_box", "x"): ("origin_span", "x", "width"),
    ("pixel_box", "y"): ("origin_span", "y", "height"),
    ("midi_tick", "tick"): ("point", "tick", None),
    ("midi_interval", "tick"): ("origin_span", "tick", "duration"),
    ("score_quarter", "quarter"): ("point", "quarter", None),
    ("score_interval", "quarter"): ("origin_span", "quarter", "duration"),
}


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
    target_field: str | None = None,
    strict: bool = False,
    reset: bool = False,
) -> BoppBase:
    """
    Trim an Annotation along a spatial, temporal, or index target axis to a range [start, end].

    Parameters
    ----------
    annotation : BoppBase
        The input Annotation model instance to trim.
    start : float or None, optional
        Start bound along the target axis. If None, no lower bound trimming is applied.
    end : float or None, optional
        End bound along the target axis. If None, no upper bound trimming is applied.
    target_field : str or None, optional
        The coordinate field along which to trim (e.g., 'time', 'x', 'y', 'tick', 'quarter', 'frequency').
        If None, the default target field for the extent type will be inferred.
        Extent types without a single canonical default (e.g. 'pixel_box') require target_field to be specified.
    strict : bool, default False
        If True, only observations entirely contained within [start, end] persist.
        If False, intervals/boxes overlapping boundary are clipped to bounds.
    reset : bool, default False
        If True, adjusts coordinate values along target_field relative to `start` (pos -> pos - start).
        Requires `start` to be non-None when True.

    Returns
    -------
    BoppBase
        A new Annotation instance with trimmed extents, payloads, and confidences.

    Raises
    ------
    BoppArgumentError
        If arguments are invalid or the extent/target_field combination is unsupported.
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
        new_parents = []
    else:
        new_parents = list(existing_parents)

    if getattr(annotation, "id", msgspec.UNSET) is msgspec.UNSET:
        # Warn that we're deriving an annotation from an unidentified annotation
        # and cannot correctly populate the parents array
        warnings.warn(
            "Trimming an annotation with no ID. The resulting annotation will have no parent lineage.",
            UserWarning,
        )
    else:
        new_parents.append(annotation.id)  # type: ignore[attr-defined]

    new_sandbox = copy.deepcopy(getattr(annotation, "sandbox", msgspec.UNSET))

    extent = getattr(annotation, "extent", msgspec.UNSET)
    if extent is msgspec.UNSET or extent is None:
        # Return copy of annotation with updated parent lineage
        new_ann = msgspec.structs.replace(
            annotation,
            id=msgspec.UNSET,
            parents=new_parents,
            sandbox=new_sandbox
        )
        validate_and_set_annotation_id(new_ann)
        return new_ann

    extent_tag = _get_tag(extent)
    if extent_tag is None:
        raise BoppArgumentError("Annotation extent object has no valid schema tag.")

    resolved_target_field = target_field
    if resolved_target_field is None:
        resolved_target_field = DEFAULT_TARGET_FIELDS.get(extent_tag)
        if resolved_target_field is None:
            raise BoppArgumentError(
                f"Extent '{extent_tag}' does not have a default target field. "
                f"Please specify 'target_field' explicitly (e.g. 'x' or 'y')."
            )

    axis_config = AXIS_CONFIGS.get((extent_tag, resolved_target_field))
    if axis_config is None:
        raise BoppArgumentError(
            f"Unsupported target_field '{resolved_target_field}' for extent tag '{extent_tag}'."
        )

    kind, field_a, field_b = axis_config

    kept_indices: list[int] = []
    shift = start if (reset and start is not None) else 0.0

    extent_updates: dict[str, list[Any]] = {}

    if kind == "point":
        pos_vals = getattr(extent, field_a)
        n_obs = len(pos_vals)
        new_pos = []

        for i, pos in enumerate(pos_vals):
            if start is not None and pos < start:
                continue
            if end is not None and pos > end:
                continue
            kept_indices.append(i)
            new_pos.append(pos - shift)

        extent_updates[field_a] = new_pos

    elif kind == "origin_span":
        origin_vals = getattr(extent, field_a)
        span_vals = getattr(extent, field_b)  # type: ignore[arg-type]
        n_obs = len(origin_vals)

        new_origin: list[float] = []
        new_span: list[float] = []

        for i, (p_min, span) in enumerate(zip(origin_vals, span_vals)):
            p_max = p_min + span

            if strict:
                if start is not None and p_min < start:
                    continue
                if end is not None and p_max > end:
                    continue
                c_min, c_max = p_min, p_max
            else:
                if start is not None and p_max <= start:
                    continue
                if end is not None and p_min >= end:
                    continue
                c_min = max(p_min, start) if start is not None else p_min
                c_max = min(p_max, end) if end is not None else p_max

            kept_indices.append(i)
            new_origin.append(c_min - shift)
            new_span.append(c_max - c_min)

        extent_updates[field_a] = new_origin
        extent_updates[field_b] = new_span  # type: ignore[index]

    elif kind == "min_max":
        min_vals = getattr(extent, field_a)
        max_vals = getattr(extent, field_b)  # type: ignore[arg-type]
        n_obs = len(min_vals)

        new_min: list[float] = []
        new_max: list[float] = []

        for i, (p_min, p_max) in enumerate(zip(min_vals, max_vals)):
            if strict:
                if start is not None and p_min < start:
                    continue
                if end is not None and p_max > end:
                    continue
                c_min, c_max = p_min, p_max
            else:
                if start is not None and p_max <= start:
                    continue
                if end is not None and p_min >= end:
                    continue
                c_min = max(p_min, start) if start is not None else p_min
                c_max = min(p_max, end) if end is not None else p_max

            kept_indices.append(i)
            new_min.append(c_min - shift)
            new_max.append(c_max - shift)

        extent_updates[field_a] = new_min
        extent_updates[field_b] = new_max  # type: ignore[index]

    # Filter remaining parallel list fields in extent
    for fname, fval in _get_list_fields_with_length(extent, n_obs).items():
        if fname not in extent_updates:
            extent_updates[fname] = [fval[idx] for idx in kept_indices]

    new_extent = msgspec.structs.replace(extent, **extent_updates)

    # Filter payload parallel fields
    payload: msgspec.Struct = annotation.payload  # type: ignore[attr-defined]
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
