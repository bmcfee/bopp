from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable
from typing import Any, Literal

import msgspec

from .base import BoppBase
from .core import validate_and_set_annotation_id
from .exceptions import BoppArgumentError, BoppValidationError
from .models.v1.extent.times import Times
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


def _derive_parents_and_sandbox(annotation: BoppBase, action_name: str) -> tuple[list[str], Any]:
    """Extract parent lineage and deep copy sandbox for a derived annotation."""
    existing_parents = getattr(annotation, "parents", None)
    if existing_parents is None or existing_parents is msgspec.UNSET:
        new_parents = []
    else:
        new_parents = list(existing_parents)

    if getattr(annotation, "id", msgspec.UNSET) is msgspec.UNSET:
        warnings.warn(
            f"{action_name} an annotation with no ID. The resulting annotation will have no parent lineage.",
            UserWarning,
        )
    else:
        new_parents.append(annotation.id)  # type: ignore[attr-defined]

    new_sandbox = copy.deepcopy(getattr(annotation, "sandbox", msgspec.UNSET))
    return new_parents, new_sandbox


def _rebuild_annotation(
    annotation: BoppBase,
    *,
    parents: list[str],
    sandbox: Any,
    payload: msgspec.Struct,
    extent: msgspec.Struct | None | Any = msgspec.UNSET,
    confidence: msgspec.Struct | None | Any = msgspec.UNSET,
) -> BoppBase:
    """Rebuild an Annotation with updated facets and recompute its deterministic ID."""
    kwargs: dict[str, Any] = {
        "id": msgspec.UNSET,
        "parents": parents,
        "sandbox": sandbox,
        "payload": payload,
    }
    if extent is not msgspec.UNSET:
        kwargs["extent"] = extent
    if confidence is not msgspec.UNSET:
        kwargs["confidence"] = confidence

    new_ann = msgspec.structs.replace(annotation, **kwargs)
    validate_and_set_annotation_id(new_ann)
    return new_ann


def _subset_struct_lists(
    struct: msgspec.Struct | None,
    indices: list[int],
    expected_length: int | None = None,
    overrides: dict[str, list[Any]] | None = None,
) -> msgspec.Struct | None:
    """Subset list fields of a struct using indices, applying optional pre-computed overrides."""
    if struct is None or struct is msgspec.UNSET:
        return struct

    updates: dict[str, list[Any]] = dict(overrides) if overrides else {}
    list_fields = (
        _get_list_fields_with_length(struct, expected_length)
        if expected_length is not None
        else _get_facet_list_fields(struct)
    )

    for fname, fval in list_fields.items():
        if fname not in updates:
            updates[fname] = [fval[idx] for idx in indices]

    return msgspec.structs.replace(struct, **updates)


def _resample_struct(
    struct: msgspec.Struct | None,
    cols: dict[str, list[Any]],
    matched_indices: list[int | None],
    fill_map: dict[str, Any],
) -> msgspec.Struct | None:
    """Resample struct list fields based on matched indices and a fill mapping."""
    if struct is None or struct is msgspec.UNSET:
        return struct

    updates: dict[str, list[Any]] = {}
    for col_name, col_values in cols.items():
        fill_val = fill_map.get(col_name)
        updates[col_name] = [
            fill_val if idx is None else col_values[idx] for idx in matched_indices
        ]

    return msgspec.structs.replace(struct, **updates)


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

    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Trimming")

    extent = getattr(annotation, "extent", msgspec.UNSET)
    payload: msgspec.Struct = annotation.payload  # type: ignore[attr-defined]
    confidence = getattr(annotation, "confidence", msgspec.UNSET)

    if extent is msgspec.UNSET or extent is None:
        return _rebuild_annotation(
            annotation,
            parents=new_parents,
            sandbox=new_sandbox,
            payload=payload,
            extent=extent,
            confidence=confidence,
        )

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

    new_extent = _subset_struct_lists(extent, kept_indices, n_obs, overrides=extent_updates)
    new_payload = _subset_struct_lists(payload, kept_indices, n_obs)
    new_confidence = _subset_struct_lists(confidence, kept_indices, n_obs)

    return _rebuild_annotation(
        annotation,
        parents=new_parents,
        sandbox=new_sandbox,
        extent=new_extent,
        payload=new_payload,  # type: ignore[arg-type]
        confidence=new_confidence,
    )


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

    >>> import bopp
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

    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Filtering")

    payload: msgspec.Struct = getattr(annotation, "payload", None)  # type: ignore[assignment]
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
        return _rebuild_annotation(
            annotation,
            parents=new_parents,
            sandbox=new_sandbox,
            payload=payload,
            extent=extent,
            confidence=confidence,
        )

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
    new_payload = _subset_struct_lists(payload, kept_indices, n_obs)
    new_extent = _subset_struct_lists(extent, kept_indices, n_obs)
    new_confidence = _subset_struct_lists(confidence, kept_indices, n_obs)

    return _rebuild_annotation(
        annotation,
        parents=new_parents,
        sandbox=new_sandbox,
        extent=new_extent,
        payload=new_payload,  # type: ignore[arg-type]
        confidence=new_confidence,
    )


def _validate_fill_values(
    struct_type: type[msgspec.Struct],
    fill_map: dict[str, Any],
    facet_name: str,
) -> None:
    """Validate that fill values conform to the struct's field types."""
    dummy_kwargs: dict[str, Any] = {}
    for f in msgspec.structs.fields(struct_type):
        if f.name in fill_map:
            dummy_kwargs[f.name] = [fill_map[f.name]]
        else:
            dummy_kwargs[f.name] = []
    try:
        msgspec.convert(dummy_kwargs, struct_type)
    except Exception as exc:
        raise BoppValidationError(
            f"Invalid fill value for {facet_name} struct '{struct_type.__name__}': {exc}"
        ) from exc


def to_times(
    annotation: BoppBase,
    *,
    times: list[float] | None = None,
    sample_rate: float | None = None,
    method: Literal["previous", "nearest"] = "previous",
    overlap: Literal["latest", "first", "multiple"] = "latest",
    fill_value: Any = None,
) -> BoppBase:
    """
    Convert an Annotation to the 'times' (point-in-time) extent via sampling or interpolation.

    Evaluates observations at the requested timestamps. For interval extents, observations
    covering each timestamp are selected. For point extents, nearest or previous neighbor
    lookup is performed without assuming any algebra on the payload values. If confidence
    values are present, they are resampled in parallel.

    Parameters
    ----------
    annotation : BoppBase
        The input Annotation instance to convert.
    times : list of float or None, optional
        Explicit array of timestamp positions against which to sample.
        Mutually exclusive with `sample_rate`.
    sample_rate : float or None, optional
        Uniform sampling rate in Hertz (> 0). Samples are generated covering the full
        extent of the annotation: from the earliest start time to the latest end time.
        Mutually exclusive with `times`.
    method : {"previous", "nearest"}, default "previous"
        Interpolation method when sampling point extents ('time'):
        - "previous": Selects the most recent prior event at or before the sample timestamp.
        - "nearest": Selects the event closest in time to the sample timestamp.
    overlap : {"latest", "first", "multiple"}, default "latest"
        Disambiguation strategy when multiple intervals cover a sample timestamp:
        - "latest": Pick the interval that starts latest (or has the highest index).
        - "first": Pick the interval that starts earliest (or has the lowest index).
        - "multiple": Emit duplicate sample timestamps, one for each overlapping interval.
    fill_value : Any or dict of str to Any, optional
        Value to fill when a sample timestamp falls in a gap (no covering observation).
        For single-column payloads, a scalar value may be supplied. For multi-column
        payloads, a dictionary mapping field names to fill values must be provided.
        Fill values are validated against the payload schema. Default is None.

    Returns
    -------
    BoppBase
        A new Annotation instance with extent type `times` (`Times` struct).

    Raises
    ------
    BoppArgumentError
        If arguments are invalid or the source extent is incompatible with time conversion.
    BoppValidationError
        If `fill_value` does not conform to the payload or confidence schema types.

    Examples
    --------
    Sample a time interval annotation at explicit timestamps:

    >>> import bopp
    >>> ann = bopp.create(
    ...     media_id="track_1",
    ...     payload_kind="tag_open",
    ...     extent_kind="time_interval",
    ...     time=[0.0, 2.0],
    ...     duration=[2.0, 2.0],
    ...     value=["verse", "chorus"],
    ... )
    >>> sampled = bopp.to_times(ann, times=[0.5, 1.5, 2.5])
    >>> sampled.extent.time
    [0.5, 1.5, 2.5]
    >>> sampled.payload.value
    ['verse', 'verse', 'chorus']

    Sample with a uniform sample rate over the full extent:

    >>> sampled_rate = bopp.to_times(ann, sample_rate=1.0)
    >>> sampled_rate.extent.time
    [0.0, 1.0, 2.0, 3.0, 4.0]
    >>> sampled_rate.payload.value
    ['verse', 'verse', 'chorus', 'chorus', 'chorus']

    Resample point events using previous-neighbor lookup:

    >>> points = bopp.create(
    ...     media_id="track_1",
    ...     payload_kind="chord",
    ...     extent_kind="time",
    ...     time=[0.0, 2.0],
    ...     value=["C:maj", "G:maj"],
    ... )
    >>> resampled = bopp.to_times(points, times=[0.5, 1.9, 2.1], method="previous")
    >>> resampled.payload.value
    ['C:maj', 'C:maj', 'G:maj']

    Handling gaps with a fill value:

    >>> intervals = bopp.create(
    ...     media_id="track_1",
    ...     payload_kind="tag_open",
    ...     extent_kind="time_interval",
    ...     time=[1.0],
    ...     duration=[1.0],
    ...     value=["solo"],
    ... )
    >>> filled = bopp.to_times(intervals, times=[0.5, 1.5, 2.5], fill_value="silence")
    >>> filled.payload.value
    ['silence', 'solo', 'silence']
    """
    if (times is None and sample_rate is None) or (times is not None and sample_rate is not None):
        raise BoppArgumentError("Exactly one of 'times' or 'sample_rate' must be specified.")

    if sample_rate is not None and sample_rate <= 0:
        raise BoppArgumentError(f"sample_rate must be positive (> 0), got {sample_rate}.")

    if method not in ("previous", "nearest"):
        raise BoppArgumentError(f"Invalid method '{method}'. Must be 'previous' or 'nearest'.")

    if overlap not in ("latest", "first", "multiple"):
        raise BoppArgumentError(
            f"Invalid overlap strategy '{overlap}'. Must be 'latest', 'first', or 'multiple'."
        )

    # Lineage tracking
    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Converting")

    extent = getattr(annotation, "extent", None)
    payload: msgspec.Struct = annotation.payload  # type: ignore[attr-defined]
    confidence = getattr(annotation, "confidence", None)

    payload_cols = _get_facet_list_fields(payload)
    confidence_cols = _get_facet_list_fields(confidence)

    # Determine extent tag & compatibility
    extent_tag: str | None = None
    if extent is not None and extent is not msgspec.UNSET:
        extent_tag = _get_tag(extent)
        if extent_tag not in ("time", "time_interval", "time_frequency_box"):
            raise BoppArgumentError(
                f"Extent type '{extent_tag}' is incompatible with to_times. "
                f"Supported extents are 'time', 'time_interval', 'time_frequency_box', or None (global)."
            )

    # Determine observations count
    n_source_obs = 0
    if extent_tag is not None and extent is not None:
        first_extent_col = next(iter(_get_facet_list_fields(extent).values()), [])
        n_source_obs = len(first_extent_col)
    elif payload_cols:
        first_payload_col = next(iter(payload_cols.values()))
        n_source_obs = len(first_payload_col)

    # Setup fill_value map
    payload_fill_map: dict[str, Any] = {}
    confidence_fill_map: dict[str, Any] = {}

    if isinstance(fill_value, dict):
        for col_name in payload_cols:
            if col_name in fill_value:
                payload_fill_map[col_name] = fill_value[col_name]
            elif "value" in fill_value and len(payload_cols) == 1:
                payload_fill_map[col_name] = fill_value["value"]
            else:
                raise BoppArgumentError(
                    f"Missing fill value for payload field '{col_name}' in fill_value dictionary."
                )
        for col_name in confidence_cols:
            if col_name in fill_value:
                confidence_fill_map[col_name] = fill_value[col_name]
            else:
                confidence_fill_map[col_name] = 0.0
    else:
        if len(payload_cols) == 1:
            col_name = next(iter(payload_cols.keys()))
            payload_fill_map[col_name] = fill_value
        elif len(payload_cols) > 1 and fill_value is not None:
            raise BoppArgumentError(
                "Multi-column payload requires fill_value to be a dict mapping field names to values."
            )
        elif len(payload_cols) > 1:
            # fill_value is None
            for col_name in payload_cols:
                payload_fill_map[col_name] = None

        for col_name in confidence_cols:
            confidence_fill_map[col_name] = 0.0

    # Build sampling grid
    target_times: list[float]
    if times is not None:
        target_times = [float(t) for t in times]
    else:
        assert sample_rate is not None
        if extent_tag is None or n_source_obs == 0:
            raise BoppArgumentError(
                "sample_rate requires an extent with observations to determine temporal bounds. "
                "Specify explicit 'times' instead."
            )
        if extent_tag == "time":
            time_arr = extent.time  # type: ignore[union-attr]
            t_min = min(time_arr)
            t_max = max(time_arr)
        elif extent_tag in ("time_interval", "time_frequency_box"):
            time_arr = extent.time  # type: ignore[union-attr]
            dur_arr = extent.duration  # type: ignore[union-attr]
            t_min = min(time_arr)
            t_max = max(t + d for t, d in zip(time_arr, dur_arr))
        else:
            raise BoppArgumentError(f"Cannot determine bounds for extent '{extent_tag}'.")

        step = 1.0 / sample_rate
        # Calculate number of steps with float tolerance
        n_steps = math.floor((t_max - t_min) / step + 1e-9) + 1
        target_times = [t_min + i * step for i in range(n_steps)]
        if target_times and target_times[-1] > t_max + 1e-9:
            target_times.pop()

    # If target_times is empty, return empty times extent
    if not target_times:
        new_extent = Times(time=[])
        new_payload = _subset_struct_lists(payload, [], n_source_obs)
        new_confidence = _subset_struct_lists(confidence, [], n_source_obs)

        return _rebuild_annotation(
            annotation,
            parents=new_parents,
            sandbox=new_sandbox,
            extent=new_extent,
            payload=new_payload,  # type: ignore[arg-type]
            confidence=new_confidence,
        )

    # If source annotation has no observations
    if n_source_obs == 0:
        # Validate fill values since every query position will be a gap
        _validate_fill_values(type(payload), payload_fill_map, "payload")
        if confidence is not None and confidence is not msgspec.UNSET:
            _validate_fill_values(type(confidence), confidence_fill_map, "confidence")

        out_times = list(target_times)
        empty_indices: list[int | None] = [None] * len(out_times)
        new_payload = _resample_struct(payload, payload_cols, empty_indices, payload_fill_map)
        new_confidence = _resample_struct(confidence, confidence_cols, empty_indices, confidence_fill_map)
        new_extent = Times(time=out_times)

        return _rebuild_annotation(
            annotation,
            parents=new_parents,
            sandbox=new_sandbox,
            extent=new_extent,
            payload=new_payload,  # type: ignore[arg-type]
            confidence=new_confidence,
        )

    out_times = []
    # matched_indices: None means gap, int means source index
    matched_indices: list[int | None] = []

    # Sampling per extent type
    if extent_tag is None:
        # Global annotation: broadcast the single observation across all query times
        for t in target_times:
            out_times.append(t)
            matched_indices.append(0)

    elif extent_tag in ("time_interval", "time_frequency_box"):
        t_starts = extent.time  # type: ignore[union-attr]
        durations = extent.duration  # type: ignore[union-attr]

        for t in target_times:
            # An interval covers t if t_start <= t <= t_start + duration
            covers = [
                i
                for i, (ts, dur) in enumerate(zip(t_starts, durations))
                if ts <= t <= ts + dur
            ]
            if not covers:
                out_times.append(t)
                matched_indices.append(None)
            elif overlap == "multiple":
                for idx in covers:
                    out_times.append(t)
                    matched_indices.append(idx)
            elif overlap == "first":
                out_times.append(t)
                matched_indices.append(covers[0])
            elif overlap == "latest":
                out_times.append(t)
                matched_indices.append(covers[-1])

    elif extent_tag == "time":
        src_times = extent.time  # type: ignore[union-attr]
        for t in target_times:
            if method == "previous":
                prev_candidates = [i for i, st in enumerate(src_times) if st <= t]
                if not prev_candidates:
                    out_times.append(t)
                    matched_indices.append(None)
                else:
                    best_idx = prev_candidates[-1]
                    out_times.append(t)
                    matched_indices.append(best_idx)
            elif method == "nearest":
                # Find index with minimum absolute distance
                best_idx = min(range(len(src_times)), key=lambda i: abs(src_times[i] - t))
                out_times.append(t)
                matched_indices.append(best_idx)

    # Validate fill_value if any gaps occurred
    if None in matched_indices:
        _validate_fill_values(type(payload), payload_fill_map, "payload")
        if confidence is not None and confidence is not msgspec.UNSET:
            _validate_fill_values(type(confidence), confidence_fill_map, "confidence")

    new_payload = _resample_struct(payload, payload_cols, matched_indices, payload_fill_map)
    new_confidence = _resample_struct(confidence, confidence_cols, matched_indices, confidence_fill_map)
    new_extent = Times(time=out_times)

    return _rebuild_annotation(
        annotation,
        parents=new_parents,
        sandbox=new_sandbox,
        extent=new_extent,
        payload=new_payload,  # type: ignore[arg-type]
        confidence=new_confidence,
    )
