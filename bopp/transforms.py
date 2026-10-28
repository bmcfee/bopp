from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable
from fractions import Fraction
from typing import Any, Literal

import msgspec

from ._version import DEFAULT_SCHEMA_VERSION
from .base import BoppBase
from .core import ensure_annotation_id
from .exceptions import BoppArgumentError, BoppValidationError
from .models.v1.extent.time import Time
from .models.v1.metadata.derived import DerivedAnnotationMetadata
from .registries import get_registry
from .util import _get_tag

# Mapping default target_field when target_field is None
DEFAULT_TARGET_FIELDS: dict[str, str] = {
    "time": "time",
    "time_interval": "time",
    "time_frequency_box": "time",
    "time_uniform": "time",
    "midi_tick": "tick",
    "midi_interval": "tick",
    "score_quarter": "quarter",
    "score_interval": "quarter",
    "score_quarter_float": "quarter",
    "score_interval_float": "quarter",
    "text": "offset",
    "text_interval": "offset",
    "spherical_time": "time",
    "spherical_time_interval": "time",
    "cartesian_time": "time",
    "cartesian_time_interval": "time",
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
    ("pixel", "x"): ("point", "x", None),
    ("pixel", "y"): ("point", "y", None),
    ("midi_tick", "tick"): ("point", "tick", None),
    ("midi_interval", "tick"): ("origin_span", "tick", "duration"),
    ("score_quarter", "quarter"): ("point", "quarter", None),
    ("score_interval", "quarter"): ("origin_span", "quarter", "duration"),
    ("score_quarter_float", "quarter"): ("point", "quarter", None),
    ("score_interval_float", "quarter"): ("origin_span", "quarter", "duration"),
    ("text", "offset"): ("point", "offset", None),
    ("text_interval", "offset"): ("origin_span", "offset", "length"),
    ("spherical", "azimuth"): ("point", "azimuth", None),
    ("spherical", "elevation"): ("point", "elevation", None),
    ("spherical", "radius"): ("point", "radius", None),
    ("spherical_time", "time"): ("point", "time", None),
    ("spherical_time", "azimuth"): ("point", "azimuth", None),
    ("spherical_time", "elevation"): ("point", "elevation", None),
    ("spherical_time", "radius"): ("point", "radius", None),
    ("spherical_time_interval", "time"): ("origin_span", "time", "duration"),
    ("spherical_time_interval", "azimuth"): ("point", "azimuth", None),
    ("spherical_time_interval", "elevation"): ("point", "elevation", None),
    ("spherical_time_interval", "radius"): ("point", "radius", None),
    ("cartesian", "x"): ("point", "x", None),
    ("cartesian", "y"): ("point", "y", None),
    ("cartesian", "z"): ("point", "z", None),
    ("cartesian_time", "time"): ("point", "time", None),
    ("cartesian_time", "x"): ("point", "x", None),
    ("cartesian_time", "y"): ("point", "y", None),
    ("cartesian_time", "z"): ("point", "z", None),
    ("cartesian_time_interval", "time"): ("origin_span", "time", "duration"),
    ("cartesian_time_interval", "x"): ("point", "x", None),
    ("cartesian_time_interval", "y"): ("point", "y", None),
    ("cartesian_time_interval", "z"): ("point", "z", None),
}

FRACTION_AXES: frozenset[tuple[str, str]] = frozenset(
    {
        ("score_quarter", "quarter"),
        ("score_interval", "quarter"),
    }
)


_SCORE_EXTENT_CONVERSIONS: dict[str, tuple[str, str]] = {
    "to_score_quarter": ("score_quarter_float", "score_quarter"),
    "to_score_interval": ("score_interval_float", "score_interval"),
    "to_score_quarter_float": ("score_quarter", "score_quarter_float"),
    "to_score_interval_float": ("score_interval", "score_interval_float"),
}


_TrimBound = int | float | Fraction | list[int] | tuple[int, int] | None


class FilterRecord(dict):
    """
    Dictionary supporting attribute access for multi-column filter predicates.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'FilterRecord' object has no attribute '{name}'"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


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


def _derive_parents_and_sandbox(
    annotation: BoppBase, action_name: str
) -> tuple[list[str], Any]:
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
    metadata: msgspec.Struct | Any = msgspec.UNSET,
    extent: msgspec.Struct | None | Any = msgspec.UNSET,
    confidence: msgspec.Struct | None | Any = msgspec.UNSET,
    groupings: list[Any] | Any = msgspec.UNSET,
) -> BoppBase:
    """Rebuild an Annotation with updated facets and recompute its deterministic ID."""
    kwargs: dict[str, Any] = {
        "id": msgspec.UNSET,
        "parents": parents,
        "sandbox": sandbox,
        "payload": payload,
    }
    if metadata is not msgspec.UNSET:
        kwargs["metadata"] = metadata
    if extent is not msgspec.UNSET:
        kwargs["extent"] = extent
    if confidence is not msgspec.UNSET:
        kwargs["confidence"] = confidence
    if groupings is not msgspec.UNSET:
        kwargs["groupings"] = groupings

    new_ann = msgspec.structs.replace(annotation, **kwargs)
    ensure_annotation_id(new_ann)
    return new_ann


def _subset_struct_lists(
    struct: msgspec.Struct | Literal[msgspec.UnsetType.UNSET] | None,
    indices: list[int],
    overrides: dict[str, list[Any]] | None = None,
) -> msgspec.Struct | Literal[msgspec.UnsetType.UNSET] | None:
    """Subset list fields of a struct using indices, applying optional pre-computed overrides."""
    if struct is None or struct is msgspec.UNSET:
        return struct

    updates: dict[str, list[Any]] = dict(overrides) if overrides else {}
    list_fields = _get_facet_list_fields(struct)

    for fname, fval in list_fields.items():
        if fname not in updates:
            updates[fname] = [fval[idx] for idx in indices]

    return msgspec.structs.replace(struct, **updates)


def _subset_groupings(
    groupings: list[Any] | Literal[msgspec.UnsetType.UNSET] | None,
    indices: list[int],
) -> list[Any] | Literal[msgspec.UnsetType.UNSET] | None:
    """Subset ids array of grouping objects using indices."""
    if groupings is None or groupings is msgspec.UNSET:
        return groupings

    new_groupings = []
    for g in groupings:
        new_ids = [g.ids[idx] for idx in indices]
        new_groupings.append(msgspec.structs.replace(g, ids=new_ids))
    return new_groupings


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


def _resample_groupings(
    groupings: list[Any] | Literal[msgspec.UnsetType.UNSET] | None,
    matched_indices: list[int | None],
    fill_map: dict[str, int],
) -> list[Any] | Literal[msgspec.UnsetType.UNSET] | None:
    """Resample groupings based on matched indices and a fill mapping."""
    if groupings is None or groupings is msgspec.UNSET:
        return groupings

    new_groupings = []
    for g in groupings:
        fill_val = fill_map.get(g.name, -1)
        new_ids = [fill_val if idx is None else g.ids[idx] for idx in matched_indices]
        new_groupings.append(msgspec.structs.replace(g, ids=new_ids))
    return new_groupings


def _coerce_trim_bound(value: _TrimBound) -> float | Fraction | None:
    """Validate and normalize rational pairs, leaving other bounds unchanged."""
    if isinstance(value, (list, tuple)):
        if (
            len(value) != 2
            or not all(type(item) is int for item in value)
            or value[1] < 1
        ):
            raise BoppArgumentError(
                "Fraction bound must be a pair of ints with denominator >= 1."
            )
        return Fraction(value[0], value[1])
    return value


def _fraction_pair(value: Fraction) -> list[int]:
    """Encode a reduced fraction as a JSON-compatible pair."""
    return [value.numerator, value.denominator]


def _conversion_extent(
    annotation: BoppBase,
    transform: str,
    max_denominator: int | None = None,
) -> BoppBase:
    """Dispatch a score conversion after validating its source and parameters."""
    source_tag, target_tag = _SCORE_EXTENT_CONVERSIONS[transform]
    extent = getattr(annotation, "extent", msgspec.UNSET)
    if extent is msgspec.UNSET or extent is None:
        raise BoppArgumentError(
            f"{transform}: annotation has no extent; requires '{source_tag}'."
        )
    tag = _get_tag(extent)
    if tag != source_tag:
        raise BoppArgumentError(
            f"{transform}: extent {tag!r} is unsupported; requires '{source_tag}'."
        )

    fraction_target = source_tag.endswith("_float")
    parameters: dict[str, Any] = {}
    limit = 1
    if fraction_target:
        if not isinstance(max_denominator, int) or isinstance(max_denominator, bool):
            raise BoppArgumentError(
                f"{transform}: max_denominator must be an int (not a bool), got {max_denominator!r}."
            )
        if max_denominator < 1:
            raise BoppArgumentError(
                f"{transform}: max_denominator must be >= 1, got {max_denominator}."
            )
        limit = max_denominator
        parameters["max_denominator"] = limit

    columns: dict[str, list[Any]] = {}
    for name, values in _get_facet_list_fields(extent).items():
        if fraction_target:
            columns[name] = []
            for index, value in enumerate(values):
                if not math.isfinite(value):
                    raise BoppArgumentError(
                        f"{transform}: field {name!r} at index {index} must be finite."
                    )
                columns[name].append(
                    _fraction_pair(Fraction(value).limit_denominator(limit))
                )
        else:
            columns[name] = [
                numerator / denominator for numerator, denominator in values
            ]
    return _rebuild_converted_extent(
        annotation, target_tag, columns, transform, parameters
    )


def _rebuild_converted_extent(
    annotation: BoppBase,
    target_tag: str,
    columns: dict[str, list[Any]],
    transform: str,
    parameters: dict[str, Any],
) -> BoppBase:
    """Build a registered twin extent and derive an annotation around it."""
    version = getattr(annotation, "bopp_version", DEFAULT_SCHEMA_VERSION)
    extent_cls = get_registry(version)["EXTENT_TYPE_REGISTRY"][target_tag]
    parents, sandbox = _derive_parents_and_sandbox(annotation, "Converting")
    extent_kwargs: dict[str, Any] = dict(columns)
    if (
        hasattr(annotation, "extent")
        and annotation.extent is not None
        and annotation.extent is not msgspec.UNSET
    ):
        extent_kwargs.setdefault(
            "disjoint", getattr(annotation.extent, "disjoint", False)
        )
        extent_kwargs.setdefault(
            "exhaustive", getattr(annotation.extent, "exhaustive", False)
        )
    return _rebuild_annotation(
        annotation,
        parents=parents,
        sandbox=sandbox,
        payload=copy.deepcopy(annotation.payload),  # type: ignore[attr-defined]
        confidence=copy.deepcopy(getattr(annotation, "confidence", msgspec.UNSET)),
        groupings=copy.deepcopy(getattr(annotation, "groupings", msgspec.UNSET)),
        extent=extent_cls(**extent_kwargs),
        metadata=DerivedAnnotationMetadata(transform=transform, parameters=parameters),
    )


def to_score_quarter(annotation: BoppBase, *, max_denominator: int) -> BoppBase:
    """
    Convert a score_quarter_float extent to score_quarter.

    Each finite coordinate in `quarter` becomes the nearest fraction
    whose denominator does not exceed the bound, using
    Fraction(value).limit_denominator(max_denominator). Approximation uses the
    float's exact binary value: 1.1 becomes [6, 5] with a bound of 5 because
    it lies slightly above the midpoint of 1 and 6/5. Results are reduced
    [numerator, denominator] lists with positive denominators.
    The source is unchanged; the derived annotation copies payload, confidence,
    and sandbox, records parent lineage and conversion metadata, and recomputes its ID.

    Parameters
    ----------
    annotation : BoppBase
        Annotation with a score_quarter_float extent.
    max_denominator : int
        Required keyword-only positive denominator bound. Booleans are not accepted.

    Returns
    -------
    BoppBase
        A derived annotation with a score_quarter extent and a new ID.

    Raises
    ------
    BoppArgumentError
        If the source extent is missing or is not score_quarter_float,
        the denominator bound is not an int >= 1, or a coordinate is not finite.
    """
    return _conversion_extent(annotation, "to_score_quarter", max_denominator)


def to_score_interval(annotation: BoppBase, *, max_denominator: int) -> BoppBase:
    """
    Convert a score_interval_float extent to score_interval.

    Each finite coordinate in `quarter` and `duration` becomes the nearest fraction
    whose denominator does not exceed the bound, using
    Fraction(value).limit_denominator(max_denominator). Approximation uses the
    float's exact binary value: 1.1 becomes [6, 5] with a bound of 5 because
    it lies slightly above the midpoint of 1 and 6/5. Results are reduced
    [numerator, denominator] lists with positive denominators.
    The source is unchanged; the derived annotation copies payload, confidence,
    and sandbox, records parent lineage and conversion metadata, and recomputes its ID.

    Parameters
    ----------
    annotation : BoppBase
        Annotation with a score_interval_float extent.
    max_denominator : int
        Required keyword-only positive denominator bound. Booleans are not accepted.

    Returns
    -------
    BoppBase
        A derived annotation with a score_interval extent and a new ID.

    Raises
    ------
    BoppArgumentError
        If the source extent is missing or is not score_interval_float,
        the denominator bound is not an int >= 1, or a coordinate is not finite.
    """
    return _conversion_extent(annotation, "to_score_interval", max_denominator)


def to_score_quarter_float(annotation: BoppBase) -> BoppBase:
    """
    Convert a score_quarter extent to score_quarter_float.

    Each [numerator, denominator] list or tuple in `quarter` becomes
    a float via true division. The result is exact when representable as a double
    and otherwise correctly rounded to the nearest double.
    The source is unchanged; the derived annotation copies payload, confidence,
    and sandbox, records parent lineage and conversion metadata, and recomputes its ID.

    Parameters
    ----------
    annotation : BoppBase
        Annotation with a score_quarter extent.

    Returns
    -------
    BoppBase
        A derived annotation with a score_quarter_float extent and a new ID.

    Raises
    ------
    BoppArgumentError
        If the source extent is missing or is not score_quarter.
    """
    return _conversion_extent(annotation, "to_score_quarter_float")


def to_score_interval_float(annotation: BoppBase) -> BoppBase:
    """
    Convert a score_interval extent to score_interval_float.

    Each [numerator, denominator] list or tuple in `quarter` and `duration` becomes
    a float via true division. The result is exact when representable as a double
    and otherwise correctly rounded to the nearest double.
    The source is unchanged; the derived annotation copies payload, confidence,
    and sandbox, records parent lineage and conversion metadata, and recomputes its ID.

    Parameters
    ----------
    annotation : BoppBase
        Annotation with a score_interval extent.

    Returns
    -------
    BoppBase
        A derived annotation with a score_interval_float extent and a new ID.

    Raises
    ------
    BoppArgumentError
        If the source extent is missing or is not score_interval.
    """
    return _conversion_extent(annotation, "to_score_interval_float")


def _trim_bound_parameter(
    value: _TrimBound,
) -> float | list[int] | None:
    """Keep numeric metadata bounds unchanged and encode rational bounds as pairs."""
    if isinstance(value, (list, tuple)):
        value = Fraction(value[0], value[1])
    return _fraction_pair(value) if isinstance(value, Fraction) else value


def trim(
    annotation: BoppBase,
    *,
    start: _TrimBound = None,
    end: _TrimBound = None,
    target_field: str | None = None,
    strict: bool = False,
    reset: bool = False,
) -> BoppBase:
    """
    Trim an Annotation along a spatial, temporal, or index target axis to a range [start, end].

    Fraction extents accept int, finite float, Fraction, or [numerator, denominator]
    list/tuple bounds (two ints, not bools, with denominator >= 1). Numeric axes and
    annotations without an extent accept only int or float bounds, allowing infinities
    but rejecting NaN. Boolean bounds are not accepted on any axis.
    Fraction axes use exact arithmetic and return reduced [numerator, denominator] lists;
    on these axes, float bounds are taken at their exact binary value via Fraction(value).

    Parameters
    ----------
    annotation : BoppBase
        The input Annotation model instance to trim.
    start : int, float, Fraction, list[int], tuple[int, int] or None, optional
        Start bound along the target axis. If None, no lower bound trimming is applied.
    end : int, float, Fraction, list[int], tuple[int, int] or None, optional
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
        A new Annotation instance with trimmed extents, payloads, confidences, and groupings.

    Raises
    ------
    BoppArgumentError
        If arguments are invalid or the extent/target_field combination is unsupported.
    """
    new_start = _coerce_trim_bound(start)
    new_end = _coerce_trim_bound(end)

    if new_start is None and new_end is None:
        raise BoppArgumentError("At least one of 'start' or 'end' must be provided.")

    if reset and new_start is None:
        raise BoppArgumentError("reset=True requires 'start' to be specified.")

    extent = getattr(annotation, "extent", msgspec.UNSET)
    axis_config = None
    fraction_axis = False
    extent_tag = None
    if extent is not msgspec.UNSET and extent is not None:
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

        if extent_tag != "time_uniform":
            axis_config = AXIS_CONFIGS.get((extent_tag, resolved_target_field))
            if axis_config is None:
                raise BoppArgumentError(
                    f"Unsupported target_field '{resolved_target_field}' for extent tag '{extent_tag}'."
                )

            fraction_axis = (extent_tag, resolved_target_field) in FRACTION_AXES
        elif resolved_target_field != "time":
            raise BoppArgumentError(
                f"Unsupported target_field '{resolved_target_field}' for extent tag 'time_uniform'."
            )

    if fraction_axis:
        for bound in (new_start, new_end):
            if bound is None or type(bound) is int or isinstance(bound, Fraction):
                continue
            if type(bound) is float:
                if not math.isfinite(bound):
                    raise BoppArgumentError("Trim bound must be finite.")
            else:
                raise BoppArgumentError(
                    "Trim bound must be an int, finite float, Fraction, or pair of ints."
                )
        new_start = Fraction(new_start) if new_start is not None else None
        new_end = Fraction(new_end) if new_end is not None else None
    elif isinstance(new_start, Fraction) or isinstance(new_end, Fraction):
        raise BoppArgumentError("Fraction or pair bounds require a fraction axis.")
    else:
        for bound in (new_start, new_end):
            if bound is None:
                continue
            if type(bound) is not int and type(bound) is not float:
                raise BoppArgumentError("Trim bound must be an int or float.")
            if type(bound) is float and math.isnan(bound):
                raise BoppArgumentError("Trim bound must not be NaN.")

    if new_start is not None and new_end is not None and new_start > new_end:
        display_start = list(start) if isinstance(start, tuple) else start
        display_end = list(end) if isinstance(end, tuple) else end
        raise BoppArgumentError(
            f"start ({display_start}) must be <= end ({display_end})."
        )

    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Trimming")
    derived_metadata = DerivedAnnotationMetadata(
        transform="trim",
        parameters={
            "start": _trim_bound_parameter(start),
            "end": _trim_bound_parameter(end),
            "target_field": target_field,
            "strict": strict,
            "reset": reset,
        },
    )

    payload: msgspec.Struct = annotation.payload  # type: ignore[attr-defined]
    confidence = getattr(annotation, "confidence", msgspec.UNSET)
    groupings = getattr(annotation, "groupings", msgspec.UNSET)

    if extent_tag == "time_uniform":
        if not isinstance(extent, msgspec.Struct):
            raise BoppValidationError(
                "Invalid or missing extent struct for 'time_uniform'."
            )
        tu_start_time = float(getattr(extent, "start_time", 0.0))
        tu_period = float(getattr(extent, "sampling_period", 0.0))

        payload_cols = _get_facet_list_fields(payload)
        confidence_cols = (
            _get_facet_list_fields(confidence)
            if confidence not in (None, msgspec.UNSET)
            else {}
        )
        n_obs = 0
        if payload_cols:
            n_obs = len(next(iter(payload_cols.values())))
        elif confidence_cols:
            n_obs = len(next(iter(confidence_cols.values())))

        tu_shift = float(new_start) if (reset and new_start is not None) else 0.0
        tu_kept_indices: list[int] = []
        for i in range(n_obs):
            t_val = tu_start_time + i * tu_period
            if new_start is not None and t_val < new_start:
                continue
            if new_end is not None and t_val > new_end:
                continue
            tu_kept_indices.append(i)

        if tu_kept_indices:
            first_idx = tu_kept_indices[0]
            new_start_time = tu_start_time + first_idx * tu_period - tu_shift
        else:
            new_start_time = (
                (tu_start_time - tu_shift)
                if new_start is None
                else (float(new_start) - tu_shift)
            )

        tu_new_extent = msgspec.structs.replace(
            extent, start_time=float(new_start_time)
        )
        new_payload = _subset_struct_lists(payload, tu_kept_indices)
        new_confidence = _subset_struct_lists(confidence, tu_kept_indices)
        new_groupings = _subset_groupings(groupings, tu_kept_indices)

        return _rebuild_annotation(
            annotation,
            parents=new_parents,
            sandbox=new_sandbox,
            metadata=derived_metadata,
            extent=tu_new_extent,
            payload=new_payload,  # type: ignore[arg-type]
            confidence=new_confidence,
            groupings=new_groupings,
        )

    if axis_config is None:
        return _rebuild_annotation(
            annotation,
            parents=new_parents,
            sandbox=new_sandbox,
            metadata=derived_metadata,
            payload=payload,
            extent=extent,
            confidence=confidence,
            groupings=groupings,
        )

    kind, field_a, field_b = axis_config

    kept_indices: list[int] = []
    shift = (
        new_start
        if (reset and new_start is not None)
        else (Fraction(0) if fraction_axis else 0.0)
    )

    extent_updates: dict[str, list[Any]] = {}

    if kind == "point":
        pos_vals = getattr(extent, field_a)
        if fraction_axis:
            pos_vals = [Fraction(*value) for value in pos_vals]
        new_pos = []

        for i, pos in enumerate(pos_vals):
            if new_start is not None and pos < new_start:
                continue
            if new_end is not None and pos > new_end:
                continue
            kept_indices.append(i)
            new_pos.append(pos - shift)

        extent_updates[field_a] = new_pos

    elif kind == "origin_span":
        origin_vals = getattr(extent, field_a)
        span_vals = getattr(extent, field_b)  # type: ignore[arg-type]
        if fraction_axis:
            origin_vals = [Fraction(*value) for value in origin_vals]
            span_vals = [Fraction(*value) for value in span_vals]

        new_origin: list[float | Fraction] = []
        new_span: list[float | Fraction] = []

        for i, (p_min, span) in enumerate(zip(origin_vals, span_vals)):
            p_max = p_min + span

            if strict:
                if new_start is not None and p_min < new_start:
                    continue
                if new_end is not None and p_max > new_end:
                    continue
                c_min, c_max = p_min, p_max
            else:
                if new_start is not None and p_max <= new_start:
                    continue
                if new_end is not None and p_min >= new_end:
                    continue
                c_min = max(p_min, new_start) if new_start is not None else p_min
                c_max = min(p_max, new_end) if new_end is not None else p_max

            kept_indices.append(i)
            new_origin.append(c_min - shift)
            new_span.append(c_max - c_min)

        extent_updates[field_a] = new_origin
        extent_updates[field_b] = new_span  # type: ignore[index]

    elif kind == "min_max":
        min_vals = getattr(extent, field_a)
        max_vals = getattr(extent, field_b)  # type: ignore[arg-type]

        new_min: list[float] = []
        new_max: list[float] = []

        for i, (p_min, p_max) in enumerate(zip(min_vals, max_vals)):
            if strict:
                if new_start is not None and p_min < new_start:
                    continue
                if new_end is not None and p_max > new_end:
                    continue
                c_min, c_max = p_min, p_max
            else:
                if new_start is not None and p_max <= new_start:
                    continue
                if new_end is not None and p_min >= new_end:
                    continue
                c_min = max(p_min, new_start) if new_start is not None else p_min
                c_max = min(p_max, new_end) if new_end is not None else p_max

            kept_indices.append(i)
            new_min.append(c_min - shift)
            new_max.append(c_max - shift)

        extent_updates[field_a] = new_min
        extent_updates[field_b] = new_max  # type: ignore[index]

    if fraction_axis:
        extent_updates = {
            field: [_fraction_pair(value) for value in values]
            for field, values in extent_updates.items()
        }

    new_extent = _subset_struct_lists(extent, kept_indices, overrides=extent_updates)
    new_payload = _subset_struct_lists(payload, kept_indices)
    new_confidence = _subset_struct_lists(confidence, kept_indices)
    new_groupings = _subset_groupings(groupings, kept_indices)

    return _rebuild_annotation(
        annotation,
        parents=new_parents,
        sandbox=new_sandbox,
        metadata=derived_metadata,
        extent=new_extent,
        payload=new_payload,  # type: ignore[arg-type]
        confidence=new_confidence,
        groupings=new_groupings,
    )


def filter_by(
    annotation: BoppBase,
    predicate: Callable[..., bool],
    *,
    target: str | None = None,
    facet: Literal["payload", "extent", "confidence", "groupings", "all"] = "payload",
) -> BoppBase:
    """
    Filter observations in an Annotation by applying a predicate function.

    Observations across extent, payload, confidence, and groupings facets are filtered
    in parallel according to their shared array indices, keeping only elements where
    `predicate` evaluates to True.

    Parameters
    ----------
    annotation : BoppBase
        The input Annotation instance to filter.
    predicate : Callable[..., bool]
        A callable taking observation elements or records and returning a truthy/falsy value.
    target : str or None, optional
        A specific field or grouping name to pass into `predicate` (e.g., 'value', 'time', 'pattern').
        If specified, `predicate` is invoked as `predicate(value)`.
        If None and `facet` is not "all", if the facet contains a single parallel column
        or a field named 'value', that column value is passed directly to `predicate`.
        Otherwise, a record (accessible via attribute and dict key) is passed to `predicate`.
    facet : {"payload", "extent", "confidence", "groupings", "all"}, default "payload"
        The facet to target when evaluating `predicate`.
        When set to "all", `predicate` is passed a row record containing fields from
        extent, payload, confidence, and groupings facets.

    Returns
    -------
    BoppBase
        A new Annotation instance containing only matching observations.

    Raises
    ------
    BoppArgumentError
        If arguments are invalid, the specified facet/target is not found, or
        the annotation has a 'time_uniform' extent.

    Examples
    --------
    Filter by payload value (default behavior):

    >>> import bopp
    >>> ann = bopp.create(
    ...     media_id="track_1",
    ...     payload_type="tag",
    ...     extent_type="time",
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
    ...     payload_type="mood_thayer",
    ...     valence=[0.5, -0.2, 0.8],
    ...     arousal=[0.1, 0.4, -0.3],
    ... )
    >>> happy = bopp.filter_by(ann_mood, lambda r: r.valence > 0 and r.arousal > 0)
    >>> happy.payload.valence
    [0.5]

    Cross-facet filtering across extent, payload, confidence, and groupings:

    >>> ann_multi = bopp.create(
    ...     media_id="track_1",
    ...     payload_type="tag",
    ...     extent_type="time",
    ...     confidence_type="likelihood",
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
    if facet not in ("payload", "extent", "confidence", "groupings", "all"):
        raise BoppArgumentError(
            f"Invalid facet '{facet}'. Must be one of 'payload', 'extent', 'confidence', 'groupings', or 'all'."
        )

    extent = getattr(annotation, "extent", None)
    if extent not in (None, msgspec.UNSET) and _get_tag(extent) == "time_uniform":
        raise BoppArgumentError(
            "filter_by is not supported for 'time_uniform' extents because non-contiguous "
            "filtering produces sampling gaps that break the uniform raster. "
            "Convert the extent first using bopp.to_times() or slice contiguously using bopp.transforms.trim()."
        )

    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Filtering")
    derived_metadata = DerivedAnnotationMetadata(
        transform="filter_by",
        parameters={"target": target, "facet": facet},
    )

    payload: msgspec.Struct = getattr(annotation, "payload", None)  # type: ignore[assignment]
    confidence = getattr(annotation, "confidence", None)
    groupings = getattr(annotation, "groupings", None)

    payload_cols = _get_facet_list_fields(payload)
    extent_cols = _get_facet_list_fields(extent)
    confidence_cols = _get_facet_list_fields(confidence)

    groupings_cols: dict[str, list[int]] = {}
    if groupings is not None and groupings is not msgspec.UNSET:
        for g in groupings:
            if g.name is not None:
                groupings_cols[g.name] = g.ids

    # Determine number of observations from any available facet column
    n_obs = 0
    all_facets_cols = [extent_cols, payload_cols, confidence_cols, groupings_cols]
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
            metadata=derived_metadata,
            payload=payload,
            extent=extent,
            confidence=confidence,
            groupings=groupings,
        )

    # Prepare input stream for predicate
    if facet == "all":
        if target is not None:
            # Look for target across all facets
            matching_cols = [
                cols[target]
                for cols in (payload_cols, extent_cols, confidence_cols, groupings_cols)
                if target in cols
            ]
            if not matching_cols:
                raise BoppArgumentError(
                    f"Target field '{target}' not found in any annotation facet."
                )
            if len(matching_cols) > 1:
                raise BoppArgumentError(
                    f"Ambiguous target field '{target}' found in multiple facets when facet='all'. "
                    "Specify 'facet' explicitly."
                )
            stream: list[Any] = matching_cols[0]
        else:
            seen_keys: set[str] = set()
            for cols in (extent_cols, payload_cols, confidence_cols, groupings_cols):
                overlap = seen_keys.intersection(cols.keys())
                if overlap:
                    raise BoppArgumentError(
                        f"Column name collision across facets for {sorted(overlap)} when facet='all'."
                    )
                seen_keys.update(cols.keys())
            stream = [
                FilterRecord(
                    {
                        **{k: v[i] for k, v in extent_cols.items()},
                        **{k: v[i] for k, v in payload_cols.items()},
                        **{k: v[i] for k, v in confidence_cols.items()},
                        **{k: v[i] for k, v in groupings_cols.items()},
                    }
                )
                for i in range(n_obs)
            ]
    elif facet == "groupings":
        if not groupings_cols:
            raise BoppArgumentError(
                "Facet 'groupings' is not present on annotation or contains no layers."
            )

        if target is not None:
            if target not in groupings_cols:
                raise BoppArgumentError(
                    f"Target grouping '{target}' not found in facet 'groupings'. Available: {list(groupings_cols.keys())}"
                )
            stream = groupings_cols[target]
        elif len(groupings_cols) == 1:
            stream = next(iter(groupings_cols.values()))
        else:
            stream = [
                FilterRecord({k: v[i] for k, v in groupings_cols.items()})
                for i in range(n_obs)
            ]
    else:
        facet_struct_map = {
            "payload": (payload, payload_cols),
            "extent": (extent, extent_cols),
            "confidence": (confidence, confidence_cols),
        }
        struct_obj, cols = facet_struct_map[facet]
        if struct_obj is None or struct_obj is msgspec.UNSET or not cols:
            raise BoppArgumentError(
                f"Facet '{facet}' is not present on annotation or contains no columns."
            )

        if target is not None:
            if target not in cols:
                raise BoppArgumentError(
                    f"Target field '{target}' not found in facet '{facet}'. Available: {list(cols.keys())}"
                )
            stream = cols[target]
        elif "value" in cols:
            stream = cols["value"]
        elif len(cols) == 1:
            stream = next(iter(cols.values()))
        else:
            # Multiple columns and no 'value' column: pass record
            stream = [
                FilterRecord({k: v[i] for k, v in cols.items()}) for i in range(n_obs)
            ]

    kept_indices = [i for i, item in enumerate(stream) if predicate(item)]

    # Update facets with kept indices
    new_payload = _subset_struct_lists(payload, kept_indices)
    new_extent = _subset_struct_lists(extent, kept_indices)
    new_confidence = _subset_struct_lists(confidence, kept_indices)
    new_groupings = _subset_groupings(groupings, kept_indices)

    return _rebuild_annotation(
        annotation,
        parents=new_parents,
        sandbox=new_sandbox,
        metadata=derived_metadata,
        extent=new_extent,
        payload=new_payload,  # type: ignore[arg-type]
        confidence=new_confidence,
        groupings=new_groupings,
    )


def _validate_fill_values(
    struct: msgspec.Struct,
    fill_map: dict[str, Any],
    facet_name: str,
) -> None:
    """Validate that fill values conform to the struct's field types."""
    struct_type = type(struct)
    dummy_kwargs: dict[str, Any] = {
        f.name: [fill_map[f.name]] if f.name in fill_map else getattr(struct, f.name)
        for f in msgspec.structs.fields(struct_type)
    }
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
    or grouping values are present, they are resampled in parallel.

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
        Grouping layers may be populated via grouping names in fill_value dict, defaulting
        to -1 when unspecified. Fill values are validated against schema types. Default is None.

    Returns
    -------
    BoppBase
        A new Annotation instance with extent type `time` (`Time` struct).

    Raises
    ------
    BoppArgumentError
        If arguments are invalid or the source extent is incompatible with time conversion.
    BoppValidationError
        If `fill_value` does not conform to the payload, confidence, or grouping schema types.

    Examples
    --------
    Sample a time interval annotation at explicit timestamps:

    >>> import bopp
    >>> ann = bopp.create(
    ...     media_id="track_1",
    ...     payload_type="tag",
    ...     extent_type="time_interval",
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
    ...     payload_type="chord",
    ...     extent_type="time",
    ...     time=[0.0, 2.0],
    ...     value=["C:maj", "G:maj"],
    ... )
    >>> resampled = bopp.to_times(points, times=[0.5, 1.9, 2.1], method="previous")
    >>> resampled.payload.value
    ['C:maj', 'C:maj', 'G:maj']

    Handling gaps with a fill value:

    >>> intervals = bopp.create(
    ...     media_id="track_1",
    ...     payload_type="tag",
    ...     extent_type="time_interval",
    ...     time=[1.0],
    ...     duration=[1.0],
    ...     value=["solo"],
    ... )
    >>> filled = bopp.to_times(intervals, times=[0.5, 1.5, 2.5], fill_value="silence")
    >>> filled.payload.value
    ['silence', 'solo', 'silence']
    """
    if (times is None and sample_rate is None) or (
        times is not None and sample_rate is not None
    ):
        raise BoppArgumentError(
            "Exactly one of 'times' or 'sample_rate' must be specified."
        )

    if sample_rate is not None and sample_rate <= 0:
        raise BoppArgumentError(
            f"sample_rate must be positive (> 0), got {sample_rate}."
        )

    if method not in ("previous", "nearest"):
        raise BoppArgumentError(
            f"Invalid method '{method}'. Must be 'previous' or 'nearest'."
        )

    if overlap not in ("latest", "first", "multiple"):
        raise BoppArgumentError(
            f"Invalid overlap strategy '{overlap}'. Must be 'latest', 'first', or 'multiple'."
        )

    # Lineage tracking
    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Converting")
    derived_metadata = DerivedAnnotationMetadata(
        transform="to_times",
        parameters={
            "sample_rate": sample_rate,
            "method": method,
            "overlap": overlap,
        },
    )

    extent = getattr(annotation, "extent", None)
    payload: msgspec.Struct = annotation.payload  # type: ignore[attr-defined]
    confidence = getattr(annotation, "confidence", None)
    groupings = getattr(annotation, "groupings", None)

    payload_cols = _get_facet_list_fields(payload)
    confidence_cols = _get_facet_list_fields(confidence)

    # Determine extent tag & compatibility
    extent_tag: str | None = None
    if extent is not None and extent is not msgspec.UNSET:
        extent_tag = _get_tag(extent)
        valid_temporal_extents = (
            "time",
            "time_interval",
            "time_frequency_box",
            "time_uniform",
            "spherical_time",
            "spherical_time_interval",
            "cartesian_time",
            "cartesian_time_interval",
        )
        if extent_tag not in valid_temporal_extents:
            raise BoppArgumentError(
                f"Extent type '{extent_tag}' is incompatible with to_times. "
                f"Supported extents are {valid_temporal_extents}, or None (global)."
            )

    # Determine observations count
    n_source_obs = 0
    if extent_tag is not None and extent is not None and extent_tag != "time_uniform":
        first_extent_col = next(iter(_get_facet_list_fields(extent).values()), [])
        n_source_obs = len(first_extent_col)
    elif payload_cols:
        first_payload_col = next(iter(payload_cols.values()))
        n_source_obs = len(first_payload_col)
    elif groupings is not None and groupings is not msgspec.UNSET:
        for g in groupings:
            g_ids = getattr(g, "ids", None)
            if isinstance(g_ids, list):
                n_source_obs = len(g_ids)
                break

    # Setup fill_value map
    payload_fill_map: dict[str, Any] = {}
    confidence_fill_map: dict[str, Any] = {}
    grouping_fill_map: dict[str, int] = {}

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
        if groupings is not None and groupings is not msgspec.UNSET:
            for g in groupings:
                g_name = getattr(g, "name", "")
                if g_name in fill_value:
                    val = fill_value[g_name]
                    if not isinstance(val, int) or isinstance(val, bool):
                        raise BoppValidationError(
                            f"Fill value for grouping '{g_name}' must be an integer, got {type(val).__name__}."
                        )
                    grouping_fill_map[g_name] = val
    else:
        if len(payload_cols) > 1 and fill_value is not None:
            raise BoppArgumentError(
                "Multi-column payload requires fill_value to be a dict mapping field names to values."
            )
        payload_fill_map = dict.fromkeys(payload_cols, fill_value)
        confidence_fill_map = dict.fromkeys(confidence_cols, 0.0)

    # Build sampling grid
    target_times: list[float]
    if times is not None:
        target_times = [float(t) for t in times]
    elif sample_rate is not None:
        if extent_tag is None or n_source_obs == 0:
            raise BoppArgumentError(
                "sample_rate requires an extent with observations to determine temporal bounds. "
                "Specify explicit 'times' instead."
            )
        if extent_tag in ("time", "spherical_time", "cartesian_time"):
            time_arr = extent.time  # type: ignore[union-attr]
            t_min = min(time_arr)
            t_max = max(time_arr)
        elif extent_tag == "time_uniform":
            st_val = float(extent.start_time)  # type: ignore[union-attr]
            sp_val = float(extent.sampling_period)  # type: ignore[union-attr]
            t_min = st_val
            t_max = st_val + max(0, n_source_obs - 1) * sp_val
        else:
            time_arr = extent.time  # type: ignore[union-attr]
            dur_arr = extent.duration  # type: ignore[union-attr]
            t_min = min(time_arr)
            t_max = max(t + d for t, d in zip(time_arr, dur_arr))

        step = 1.0 / sample_rate
        # Calculate number of steps with float tolerance
        n_steps = math.floor((t_max - t_min) / step + 1e-9) + 1
        target_times = [t_min + i * step for i in range(n_steps)]

    # If target_times is empty, return empty times extent
    if not target_times:
        new_extent = Time(time=[], disjoint=True, exhaustive=False)
        new_payload = _subset_struct_lists(payload, [])
        new_confidence = _subset_struct_lists(confidence, [])
        new_groupings = _subset_groupings(groupings, [])

        return _rebuild_annotation(
            annotation,
            parents=new_parents,
            sandbox=new_sandbox,
            metadata=derived_metadata,
            extent=new_extent,
            payload=new_payload,  # type: ignore[arg-type]
            confidence=new_confidence,
            groupings=new_groupings,
        )

    out_times: list[float] = []
    # matched_indices: None means gap, int means source index
    matched_indices: list[int | None] = []

    if n_source_obs == 0:
        out_times = list(target_times)
        matched_indices = [None] * len(out_times)
    elif extent_tag is None:
        # Global annotation: broadcast the single observation across all query times
        out_times = list(target_times)
        matched_indices = [0] * len(out_times)
    elif extent_tag in (
        "time_interval",
        "time_frequency_box",
        "spherical_time_interval",
        "cartesian_time_interval",
    ):
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

    elif extent_tag in (
        "time",
        "spherical_time",
        "cartesian_time",
        "time_uniform",
    ):
        if extent_tag == "time_uniform":
            st_val = float(extent.start_time)  # type: ignore[union-attr]
            sp_val = float(extent.sampling_period)  # type: ignore[union-attr]
            src_times = [st_val + i * sp_val for i in range(n_source_obs)]
        else:
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
                best_idx = min(
                    range(len(src_times)), key=lambda i: abs(src_times[i] - t)
                )
                out_times.append(t)
                matched_indices.append(best_idx)

    # Validate fill_value if any gaps occurred
    if None in matched_indices:
        _validate_fill_values(payload, payload_fill_map, "payload")
        if confidence is not None and confidence is not msgspec.UNSET:
            _validate_fill_values(confidence, confidence_fill_map, "confidence")

    new_payload = _resample_struct(
        payload, payload_cols, matched_indices, payload_fill_map
    )
    new_confidence = _resample_struct(
        confidence, confidence_cols, matched_indices, confidence_fill_map
    )
    new_groupings = _resample_groupings(groupings, matched_indices, grouping_fill_map)
    is_disj = len(set(out_times)) == len(out_times)
    new_extent = Time(time=out_times, disjoint=is_disj, exhaustive=False)

    return _rebuild_annotation(
        annotation,
        parents=new_parents,
        sandbox=new_sandbox,
        metadata=derived_metadata,
        extent=new_extent,
        payload=new_payload,  # type: ignore[arg-type]
        confidence=new_confidence,
        groupings=new_groupings,
    )
