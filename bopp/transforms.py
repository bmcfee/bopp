from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from typing import Any, Literal

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


class FilterRecord(dict):
    """
    Dictionary supporting attribute access for multi-column filter predicates.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'FilterRecord' object has no attribute '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _get_list_fields_with_length(struct: msgspec.Struct, expected_length: int) -> dict[str, list[Any]]:
    """Return dictionary of field name -> list value for list fields matching expected_length."""
    result = {}
    for field in msgspec.structs.fields(type(struct)):
        val = getattr(struct, field.name)
        if isinstance(val, list) and len(val) == expected_length:
            result[field.name] = val
    return result


def _get_facet_list_fields(struct: msgspec.Struct | None) -> dict[str, list[Any]]:
    """Return dictionary of field name -> list value for all list fields in a struct."""
    if struct is None or struct is msgspec.UNSET:
        return {}
    result = {}
    for field in msgspec.structs.fields(type(struct)):
        val = getattr(struct, field.name)
        if isinstance(val, list):
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


def filter_by(
    annotation: BoppBase,
    predicate: Callable[..., bool],
    *,
    target: str | None = None,
    facet: Literal["payload", "extent", "confidence", "all"] = "payload",
) -> BoppBase:
    """
    Filter observations in an Annotation by applying a predicate function.

    Observations across extent, payload, and confidence facets are filtered in parallel
    according to their shared array indices, keeping only elements where `predicate`
    evaluates to True.

    Parameters
    ----------
    annotation : BoppBase
        The input Annotation instance to filter.
    predicate : Callable[..., bool]
        A callable taking observation elements or records and returning a truthy/falsy value.
    target : str or None, optional
        A specific field name to pass into `predicate` (e.g., 'value', 'time', 'valence').
        If specified, `predicate` is invoked as `predicate(value)`.
        If None and `facet` is not "all", if the facet contains a single parallel column
        or a field named 'value', that column value is passed directly to `predicate`.
        Otherwise, a record (accessible via attribute and dict key) is passed to `predicate`.
    facet : {"payload", "extent", "confidence", "all"}, default "payload"
        The facet to target when evaluating `predicate`.
        When set to "all", `predicate` is passed a row record containing fields from
        extent, payload, and confidence facets.

    Returns
    -------
    BoppBase
        A new Annotation instance containing only matching observations.

    Raises
    ------
    BoppArgumentError
        If arguments are invalid or the specified facet/target is not found.

    Examples
    --------
    Filter by payload value (default behavior):

    >>> ann = bopp.create(
    ...     media_id="track_1",
    ...     payload_kind="tag_open",
    ...     extent_kind="time",
    ...     time=[1.0, 2.0, 3.0],
    ...     value=["rock", "pop", "rock"],
    ... )
    >>> filtered = bopp.filter_by(ann, lambda v: v == "rock")
    >>> filtered.payload.value
    ['rock', 'rock']
    >>> filtered.extent.time
    [1.0, 3.0]

    Filter by an explicit extent field:

    >>> filtered = bopp.filter_by(ann, lambda t: t > 1.5, facet="extent", target="time")
    >>> filtered.extent.time
    [2.0, 3.0]

    Filter multi-column payloads using record attribute access:

    >>> ann_mood = bopp.create(
    ...     media_id="track_1",
    ...     payload_kind="mood_thayer",
    ...     valence=[0.5, -0.2, 0.8],
    ...     arousal=[0.1, 0.4, -0.3],
    ... )
    >>> happy = bopp.filter_by(ann_mood, lambda r: r.valence > 0 and r.arousal > 0)
    >>> happy.payload.valence
    [0.5]

    Cross-facet filtering across extent, payload, and confidence:

    >>> ann_multi = bopp.create(
    ...     media_id="track_1",
    ...     payload_kind="tag_open",
    ...     extent_kind="time",
    ...     confidence_kind="likelihood",
    ...     time=[1.0, 2.0, 3.0],
    ...     value=["intro", "verse", "chorus"],
    ...     confidence=[0.9, 0.4, 0.95],
    ... )
    >>> res = bopp.filter_by(
    ...     ann_multi,
    ...     lambda r: r.time >= 2.0 and r.confidence >= 0.8,
    ...     facet="all",
    ... )
    >>> res.payload.value
    ['chorus']
    """
    if facet not in ("payload", "extent", "confidence", "all"):
        raise BoppArgumentError(
            f"Invalid facet '{facet}'. Must be one of 'payload', 'extent', 'confidence', or 'all'."
        )

    # Accumulate parent IDs
    existing_parents = getattr(annotation, "parents", None)
    if existing_parents is None or existing_parents is msgspec.UNSET:
        new_parents = []
    else:
        new_parents = list(existing_parents)

    if getattr(annotation, "id", msgspec.UNSET) is msgspec.UNSET:
        warnings.warn(
            "Filtering an annotation with no ID. The resulting annotation will have no parent lineage.",
            UserWarning,
        )
    else:
        new_parents.append(annotation.id)  # type: ignore[attr-defined]

    new_sandbox = copy.deepcopy(getattr(annotation, "sandbox", msgspec.UNSET))

    payload = getattr(annotation, "payload", None)
    extent = getattr(annotation, "extent", None)
    confidence = getattr(annotation, "confidence", None)

    payload_cols = _get_facet_list_fields(payload)
    extent_cols = _get_facet_list_fields(extent)
    confidence_cols = _get_facet_list_fields(confidence)

    # Determine number of observations from any available facet column
    n_obs = 0
    all_facets_cols = [payload_cols, extent_cols, confidence_cols]
    for cols in all_facets_cols:
        if cols:
            first_col = next(iter(cols.values()))
            n_obs = len(first_col)
            break

    if n_obs == 0:
        new_ann = msgspec.structs.replace(
            annotation,
            id=msgspec.UNSET,
            parents=new_parents,
            sandbox=new_sandbox,
        )
        validate_and_set_annotation_id(new_ann)
        return new_ann

    # Prepare input stream for predicate
    kept_indices: list[int] = []

    if facet == "all":
        if target is not None:
            # Look for target across all facets
            found_col = None
            for cols in (payload_cols, extent_cols, confidence_cols):
                if target in cols:
                    found_col = cols[target]
                    break
            if found_col is None:
                raise BoppArgumentError(f"Target field '{target}' not found in any annotation facet.")
            for i, val in enumerate(found_col):
                if predicate(val):
                    kept_indices.append(i)
        else:
            for i in range(n_obs):
                record_dict: dict[str, Any] = {}
                for k, v in extent_cols.items():
                    record_dict[k] = v[i]
                for k, v in payload_cols.items():
                    record_dict[k] = v[i]
                for k, v in confidence_cols.items():
                    record_dict[k] = v[i]
                record = FilterRecord(record_dict)
                if predicate(record):
                    kept_indices.append(i)

    else:
        facet_struct_map = {
            "payload": (payload, payload_cols),
            "extent": (extent, extent_cols),
            "confidence": (confidence, confidence_cols),
        }
        struct_obj, cols = facet_struct_map[facet]
        if struct_obj is None or struct_obj is msgspec.UNSET or not cols:
            raise BoppArgumentError(f"Facet '{facet}' is not present on annotation or contains no columns.")

        if target is not None:
            if target not in cols:
                raise BoppArgumentError(
                    f"Target field '{target}' not found in facet '{facet}'. Available: {list(cols.keys())}"
                )
            target_list = cols[target]
            for i, val in enumerate(target_list):
                if predicate(val):
                    kept_indices.append(i)
        else:
            # If there's a 'value' column or only 1 column, pass values directly
            if "value" in cols:
                target_list = cols["value"]
                for i, val in enumerate(target_list):
                    if predicate(val):
                        kept_indices.append(i)
            elif len(cols) == 1:
                target_list = next(iter(cols.values()))
                for i, val in enumerate(target_list):
                    if predicate(val):
                        kept_indices.append(i)
            else:
                # Multiple columns and no 'value' column: pass record
                for i in range(n_obs):
                    record = FilterRecord({k: v[i] for k, v in cols.items()})
                    if predicate(record):
                        kept_indices.append(i)

    # Update facets with kept indices
    kwargs: dict[str, Any] = {}

    if payload is not None and payload is not msgspec.UNSET:
        payload_updates = {
            fname: [fval[idx] for idx in kept_indices]
            for fname, fval in _get_list_fields_with_length(payload, n_obs).items()
        }
        kwargs["payload"] = msgspec.structs.replace(payload, **payload_updates)

    if extent is not None and extent is not msgspec.UNSET:
        extent_updates = {
            fname: [fval[idx] for idx in kept_indices]
            for fname, fval in _get_list_fields_with_length(extent, n_obs).items()
        }
        kwargs["extent"] = msgspec.structs.replace(extent, **extent_updates)

    if confidence is not None and confidence is not msgspec.UNSET:
        confidence_updates = {
            fname: [fval[idx] for idx in kept_indices]
            for fname, fval in _get_list_fields_with_length(confidence, n_obs).items()
        }
        kwargs["confidence"] = msgspec.structs.replace(confidence, **confidence_updates)

    new_ann = msgspec.structs.replace(
        annotation,
        id=msgspec.UNSET,
        parents=new_parents,
        sandbox=new_sandbox,
        **kwargs,
    )
    validate_and_set_annotation_id(new_ann)
    return new_ann
