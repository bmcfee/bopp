from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, Literal, cast

import msgspec

from ._version import DEFAULT_SCHEMA_VERSION
from .base import BoppBase
from .core import ensure_annotation_id
from .exceptions import BoppArgumentError, BoppValidationError
from .registries import get_registry

_DEFAULT_REGISTRY = get_registry(DEFAULT_SCHEMA_VERSION)
Pixel = _DEFAULT_REGISTRY["EXTENT_TYPE_REGISTRY"]["pixel"]
Time = _DEFAULT_REGISTRY["EXTENT_TYPE_REGISTRY"]["time"]
UniformTimeExtent = _DEFAULT_REGISTRY["EXTENT_TYPE_REGISTRY"]["time_uniform"]
DerivedAnnotationMetadata = _DEFAULT_REGISTRY["METADATA_TYPE_REGISTRY"]["derived"]
from .util import (
    EXTENT_AXES_METADATA,
    FRACTION_EXTENT_TAGS,
    _get_tag,
    _to_coord,
    get_extent_bounds,
    is_disjoint,
    is_exhaustive,
)

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

INTEGER_AXES: frozenset[tuple[str, str]] = frozenset(
    {
        ("text", "offset"),
        ("text_interval", "offset"),
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
    for field_name in struct.__struct_fields__:
        if field_name.startswith(("min_", "max_")) or field_name in (
            "disjoint",
            "exhaustive",
        ):
            continue
        val = getattr(struct, field_name)
        if isinstance(val, list):
            result[field_name] = val
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
        new_groupings.append(msgspec.structs.replace(g, ids=new_ids, exhaustive=False))
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


def _get_extent_axis_bound_attrs(extent_tag: str, target_field: str) -> tuple[str, str]:
    """Return the (min_attr, max_attr) field names for an extent axis."""
    specs = EXTENT_AXES_METADATA.get(extent_tag, [])
    for spec in specs:
        if (
            spec.get("origin") == target_field
            or spec.get("field") == target_field
            or spec.get("dim") == target_field
        ):
            return spec["min_attr"], spec["max_attr"]
    raise BoppArgumentError(
        f"No bounds metadata found for extent '{extent_tag}' and field '{target_field}'."
    )


def _format_bound_value(val: Any, *, fraction_axis: bool, integer_axis: bool) -> Any:
    """Format a coordinate bound value into the schema-conforming type for the axis."""
    if fraction_axis:
        if isinstance(val, (list, tuple)):
            return [int(val[0]), int(val[1])]
        if isinstance(val, Fraction):
            return _fraction_pair(val)
        return _fraction_pair(Fraction(val))
    if integer_axis:
        return round(val)
    return float(val)


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
    old_extent = getattr(annotation, "extent", None)
    extent_kwargs.setdefault("disjoint", getattr(old_extent, "disjoint", False))
    extent_kwargs.setdefault("exhaustive", getattr(old_extent, "exhaustive", False))
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
    integer_axis = False
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
            integer_axis = (extent_tag, resolved_target_field) in INTEGER_AXES
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
        tu_extent = cast(msgspec.Struct, extent)
        tu_start_time = float(getattr(extent, "start_time", 0.0))
        tu_period = float(getattr(extent, "sampling_period", 0.0))

        payload_cols = _get_facet_list_fields(payload)
        confidence_cols = _get_facet_list_fields(
            confidence if confidence is not msgspec.UNSET else None
        )
        n_obs = len(next(iter((payload_cols or confidence_cols).values()), []))

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

        tu_updates: dict[str, Any] = {"start_time": float(new_start_time)}
        old_min = getattr(extent, "min_time", msgspec.UNSET)
        old_max = getattr(extent, "max_time", msgspec.UNSET)
        old_min_f = (
            float(old_min)
            if old_min is not msgspec.UNSET and old_min is not None
            else None
        )
        old_max_f = (
            float(old_max)
            if old_max is not msgspec.UNSET and old_max is not None
            else None
        )

        if old_min_f is not None and old_max_f is not None:
            coord_min = (
                max(old_min_f, float(new_start)) if new_start is not None else old_min_f
            )
            coord_max = (
                min(old_max_f, float(new_end)) if new_end is not None else old_max_f
            )
            coord_max = max(coord_max, coord_min)
            tu_updates["min_time"] = coord_min - tu_shift if reset else coord_min
            tu_updates["max_time"] = coord_max - tu_shift if reset else coord_max
        elif old_min_f is not None:
            coord_min = (
                max(old_min_f, float(new_start)) if new_start is not None else old_min_f
            )
            tu_updates["min_time"] = coord_min - tu_shift if reset else coord_min
        elif old_max_f is not None:
            coord_max = (
                min(old_max_f, float(new_end)) if new_end is not None else old_max_f
            )
            tu_updates["max_time"] = coord_max - tu_shift if reset else coord_max

        tu_exhaustive = False
        if (
            getattr(extent, "exhaustive", False) is True
            and len(tu_kept_indices) > 0
            and not strict
        ):
            candidate_tu = msgspec.structs.replace(
                tu_extent, exhaustive=False, **tu_updates
            )
            if is_exhaustive(candidate_tu, N=len(tu_kept_indices)):
                tu_exhaustive = True

        tu_updates["exhaustive"] = tu_exhaustive
        tu_new_extent = msgspec.structs.replace(tu_extent, **tu_updates)
        new_payload = _subset_struct_lists(payload, tu_kept_indices)
        new_confidence = _subset_struct_lists(confidence, tu_kept_indices)
        new_groupings = _subset_groupings(groupings, tu_kept_indices)
        if new_groupings not in (None, msgspec.UNSET):
            updated_groupings = []
            for g in new_groupings:
                g_exhaustive = bool(
                    getattr(g, "exhaustive", False) is True
                    and len(tu_kept_indices) > 0
                    and not strict
                    and is_exhaustive(
                        tu_new_extent,
                        grouping=g,
                        N=len(tu_kept_indices),
                    )
                )
                updated_groupings.append(
                    msgspec.structs.replace(g, exhaustive=g_exhaustive)
                )
            new_groupings = updated_groupings

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

    if axis_config is None or not isinstance(extent, msgspec.Struct):
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
        else (Fraction(0) if fraction_axis else (0 if integer_axis else 0.0))
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

    else:  # kind == "min_max"
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
    elif integer_axis:
        extent_updates = {
            field: [round(value) for value in values]
            for field, values in extent_updates.items()
        }

    min_max_attrs = (
        _get_extent_axis_bound_attrs(extent_tag, resolved_target_field)
        if extent_tag is not None and resolved_target_field is not None
        else None
    )
    scalar_updates: dict[str, Any] = {}
    if min_max_attrs is not None:
        min_attr, max_attr = min_max_attrs
        old_min = getattr(extent, min_attr, msgspec.UNSET)
        old_max = getattr(extent, max_attr, msgspec.UNSET)
        has_old_min = old_min is not msgspec.UNSET and old_min is not None
        has_old_max = old_max is not msgspec.UNSET and old_max is not None

        if has_old_min and has_old_max:
            coord_old_min = _to_coord(old_min)
            coord_old_max = _to_coord(old_max)
            coord_min = (
                max(coord_old_min, new_start)
                if new_start is not None
                else coord_old_min
            )
            coord_max = (
                min(coord_old_max, new_end) if new_end is not None else coord_old_max
            )
            coord_max = max(coord_max, coord_min)
            final_min = coord_min - shift if reset else coord_min
            final_max = coord_max - shift if reset else coord_max
            scalar_updates[min_attr] = _format_bound_value(
                final_min, fraction_axis=fraction_axis, integer_axis=integer_axis
            )
            scalar_updates[max_attr] = _format_bound_value(
                final_max, fraction_axis=fraction_axis, integer_axis=integer_axis
            )
        elif has_old_min:
            coord_old_min = _to_coord(old_min)
            coord_min = (
                max(coord_old_min, new_start)
                if new_start is not None
                else coord_old_min
            )
            final_min = coord_min - shift if reset else coord_min
            scalar_updates[min_attr] = _format_bound_value(
                final_min, fraction_axis=fraction_axis, integer_axis=integer_axis
            )
        elif has_old_max:
            coord_old_max = _to_coord(old_max)
            coord_max = (
                min(coord_old_max, new_end) if new_end is not None else coord_old_max
            )
            final_max = coord_max - shift if reset else coord_max
            scalar_updates[max_attr] = _format_bound_value(
                final_max, fraction_axis=fraction_axis, integer_axis=integer_axis
            )

    raw_extent = _subset_struct_lists(extent, kept_indices, overrides=extent_updates)
    candidate_extent: msgspec.Struct = (
        raw_extent if isinstance(raw_extent, msgspec.Struct) else extent
    )

    if scalar_updates:
        candidate_extent = msgspec.structs.replace(candidate_extent, **scalar_updates)

    extent_exhaustive = bool(
        getattr(extent, "exhaustive", False) is True
        and len(kept_indices) > 0
        and not strict
        and is_exhaustive(candidate_extent, N=len(kept_indices))
    )

    scalar_updates["exhaustive"] = extent_exhaustive
    new_extent = msgspec.structs.replace(candidate_extent, **scalar_updates)
    new_payload = _subset_struct_lists(payload, kept_indices)
    new_confidence = _subset_struct_lists(confidence, kept_indices)
    new_groupings = _subset_groupings(groupings, kept_indices)
    if new_groupings not in (None, msgspec.UNSET):
        updated_groupings = []
        for g in new_groupings:
            g_exhaustive = bool(
                getattr(g, "exhaustive", False) is True
                and len(kept_indices) > 0
                and not strict
                and is_exhaustive(
                    new_extent,
                    grouping=g,
                    N=len(kept_indices),
                )
            )
            updated_groupings.append(
                msgspec.structs.replace(g, exhaustive=g_exhaustive)
            )
        new_groupings = updated_groupings

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
            groupings_cols[g.name] = g.ids

    source_cols = payload_cols or extent_cols or confidence_cols or groupings_cols
    n_obs = len(next(iter(source_cols.values()), []))

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

    if (
        new_extent is not None
        and new_extent is not msgspec.UNSET
        and getattr(new_extent, "exhaustive", False) is True
        and len(kept_indices) < n_obs
    ):
        new_extent = msgspec.structs.replace(new_extent, exhaustive=False)

    if (
        new_groupings is not None
        and new_groupings is not msgspec.UNSET
        and len(kept_indices) < n_obs
    ):
        new_groupings = [
            (
                msgspec.structs.replace(g, exhaustive=False)
                if getattr(g, "exhaustive", False) is True
                else g
            )
            for g in new_groupings
        ]

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
        fname: [fill_map[fname]] if fname in fill_map else getattr(struct, fname)
        for fname in struct_type.__struct_fields__
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
    else:
        source_cols = payload_cols or confidence_cols
        n_source_obs = len(next(iter(source_cols.values()), []))
        if not source_cols and groupings not in (None, msgspec.UNSET):
            n_source_obs = len(groupings[0].ids) if groupings else 0

    # Setup fill_value map
    payload_fill_map: dict[str, Any] = {}
    confidence_fill_map: dict[str, Any] = {}
    grouping_fill_map: dict[str, int] = {}

    if isinstance(fill_value, dict):
        for col_name in payload_cols:
            if col_name in fill_value:
                payload_fill_map[col_name] = fill_value[col_name]
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
    else:  # sample_rate is not None
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

        step = 1.0 / cast(float, sample_rate)
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
        all_indices = list(range(n_source_obs))
        for t in target_times:
            if overlap == "multiple":
                for idx in all_indices:
                    out_times.append(t)
                    matched_indices.append(idx)
            elif overlap == "first":
                out_times.append(t)
                matched_indices.append(all_indices[0])
            else:  # overlap == "latest"
                out_times.append(t)
                matched_indices.append(all_indices[-1])
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
                if ts - 1e-9 <= t <= ts + dur + 1e-9
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
                matched_indices.append(min(covers, key=lambda i: t_starts[i]))
            else:  # overlap == "latest"
                out_times.append(t)
                matched_indices.append(max(covers, key=lambda i: t_starts[i]))

    else:
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
            else:  # method == "nearest"
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

    if new_groupings not in (None, msgspec.UNSET):
        new_groupings = [
            msgspec.structs.replace(g, disjoint=is_disjoint(new_extent, grouping=g))
            for g in new_groupings
        ]

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


def sort(
    annotation: BoppBase,
    by: str | Sequence[str],
    *,
    ascending: bool | Sequence[bool] = True,
) -> BoppBase:
    """
    Sort observations in an Annotation by one or more facet fields or grouping layers.

    Observations across extent, payload, confidence, and groupings facets are reordered
    in parallel according to the specified sort keys and ordering conventions. Tie-breaking
    is performed using subsequent keys in `by`. Fraction coordinates (e.g. on
    'score_quarter' or 'score_interval' extents) are evaluated by numerical fractional
    value rather than array elements.

    Parameters
    ----------
    annotation : BoppBase
        The input Annotation instance to sort.
    by : str or Sequence of str
        Field name(s) or grouping identifier(s) to sort by. Can be:
        - A bare field name (e.g., 'value', 'time', 'confidence') if unambiguous.
        - A qualified facet field (e.g., 'payload.value', 'extent.time', 'confidence.likelihood').
        - 'grouping' or 'groupings' if the annotation has exactly one grouping layer.
        - 'grouping.N' or 'group.N' where N is the integer index of the grouping layer.
        - 'grouping.<name>' or 'group.<name>' or bare '<name>' for named grouping layers.
        - A list or sequence of the above to break ties.
    ascending : bool or Sequence of bool, default True
        Sort order. If a single bool, applies to all keys in `by`. If a sequence of bools,
        must match the length of `by`. True sorts in ascending order; False in descending order.

    Returns
    -------
    BoppBase
        A new derived Annotation instance with reordered observations and updated parent lineage.

    Raises
    ------
    BoppArgumentError
        If arguments are invalid, keys are ambiguous or not found, or the annotation
        has a 'time_uniform' extent.

    Examples
    --------
    Sort observations by payload value ascending:

    >>> import bopp
    >>> ann = bopp.create(
    ...     media_id="track_1",
    ...     payload_type="tag",
    ...     extent_type="time",
    ...     time=[3.0, 1.0, 2.0],
    ...     value=["c", "a", "b"],
    ... )
    >>> sorted_ann = bopp.transforms.sort(ann, by="value")
    >>> sorted_ann.payload.value
    ['a', 'b', 'c']
    >>> sorted_ann.extent.time
    [1.0, 2.0, 3.0]

    Sort by extent time descending:

    >>> sorted_desc = bopp.transforms.sort(ann, by="extent.time", ascending=False)
    >>> sorted_desc.extent.time
    [3.0, 2.0, 1.0]

    Sort by grouping layer with tie-breaking on extent time:

    >>> ann_multi = bopp.create(
    ...     media_id="track_1",
    ...     payload_type="tag",
    ...     extent_type="time",
    ...     time=[1.5, 0.5, 2.0, 1.0],
    ...     value=["a", "b", "c", "d"],
    ...     groupings=[{"name": "section", "ids": [1, 0, 1, 0]}],
    ... )
    >>> res = bopp.transforms.sort(ann_multi, by=["grouping.0", "extent.time"])
    >>> res.groupings[0].ids
    [0, 0, 1, 1]
    >>> res.extent.time
    [0.5, 1.0, 1.5, 2.0]
    """
    extent = getattr(annotation, "extent", None)
    if extent not in (None, msgspec.UNSET) and _get_tag(extent) == "time_uniform":
        raise BoppArgumentError(
            "sort is not supported for 'time_uniform' extents because reordering observations "
            "breaks the uniform raster. Convert the extent first using bopp.to_times()."
        )

    if isinstance(by, str):
        by_list = [by]
    elif isinstance(by, Sequence) and not isinstance(by, (str, bytes)):
        by_list = list(by)
        if not by_list:
            raise BoppArgumentError("'by' must not be an empty sequence.")
    else:
        raise BoppArgumentError(
            f"'by' must be a str or Sequence[str], got {type(by).__name__}."
        )

    for k in by_list:
        if not isinstance(k, str):
            raise BoppArgumentError(
                f"Sort key must be a string, got {type(k).__name__}."
            )

    if isinstance(ascending, bool):
        asc_list = [ascending] * len(by_list)
    elif isinstance(ascending, Sequence) and not isinstance(ascending, (str, bytes)):
        asc_list = list(ascending)
        if len(asc_list) != len(by_list):
            raise BoppArgumentError(
                f"Length of 'ascending' ({len(asc_list)}) must match length of 'by' ({len(by_list)})."
            )
        for a in asc_list:
            if type(a) is not bool:
                raise BoppArgumentError(
                    f"'ascending' elements must be booleans, got {type(a).__name__}."
                )
    else:
        raise BoppArgumentError(
            f"'ascending' must be a bool or Sequence[bool], got {type(ascending).__name__}."
        )

    payload: msgspec.Struct = getattr(annotation, "payload", None)  # type: ignore[assignment]
    confidence = getattr(annotation, "confidence", None)
    groupings = getattr(annotation, "groupings", None)

    payload_cols = _get_facet_list_fields(payload)
    extent_cols = _get_facet_list_fields(extent)
    if extent not in (None, msgspec.UNSET) and _get_tag(extent) in FRACTION_EXTENT_TAGS:
        extent_cols = {
            name: [_to_coord(value) for value in values]
            for name, values in extent_cols.items()
        }
    confidence_cols = _get_facet_list_fields(confidence)

    groupings_list: list[Any] = []
    groupings_cols: dict[str, list[int]] = {}
    if groupings not in (None, msgspec.UNSET):
        groupings_list = list(groupings)
        for g in groupings_list:
            groupings_cols[g.name] = g.ids

    source_cols = payload_cols or extent_cols or confidence_cols or groupings_cols
    n_obs = len(next(iter(source_cols.values()), []))

    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Sorting")
    derived_metadata = DerivedAnnotationMetadata(
        transform="sort",
        parameters={
            "by": by if isinstance(by, list) else list(by_list),
            "ascending": (
                ascending
                if isinstance(ascending, list)
                else (
                    asc_list[0]
                    if len(asc_list) == 1 and isinstance(ascending, bool)
                    else asc_list
                )
            ),
        },
    )

    def _resolve_sort_key(key: str) -> list[Any]:
        if key in ("grouping", "groupings"):
            if not groupings_list:
                raise BoppArgumentError(
                    "Sort key 'grouping' requested, but annotation has no groupings."
                )
            if len(groupings_list) > 1:
                raise BoppArgumentError(
                    f"Sort key '{key}' is ambiguous because annotation has {len(groupings_list)} groupings. "
                    "Specify grouping layer by index (e.g., 'grouping.0') or by name (e.g., 'grouping.<name>')."
                )
            return groupings_list[0].ids

        if key.startswith(("grouping.", "groupings.", "group.")):
            suffix = key.split(".", 1)[1]
            if suffix.isdigit():
                idx = int(suffix)
                if idx < 0 or idx >= len(groupings_list):
                    raise BoppArgumentError(
                        f"Grouping index {idx} out of range (annotation has {len(groupings_list)} groupings)."
                    )
                return groupings_list[idx].ids
            for g in groupings_list:
                if g.name == suffix:
                    return g.ids
            raise BoppArgumentError(
                f"Grouping layer '{suffix}' not found in annotation groupings."
            )

        if "." in key:
            facet_part, field_part = key.split(".", 1)
            if facet_part == "payload":
                if field_part not in payload_cols:
                    raise BoppArgumentError(
                        f"Field '{field_part}' not found in payload columns: {list(payload_cols.keys())}."
                    )
                return payload_cols[field_part]
            elif facet_part == "extent":
                if field_part not in extent_cols:
                    raise BoppArgumentError(
                        f"Field '{field_part}' not found in extent columns: {list(extent_cols.keys())}."
                    )
                vals = extent_cols[field_part]
                extent_tag = _get_tag(cast(msgspec.Struct, extent))
                if (extent_tag, field_part) in FRACTION_AXES:
                    return [
                        (Fraction(*v) if isinstance(v, (list, tuple)) else Fraction(v))
                        for v in vals
                    ]
                return vals
            elif facet_part == "confidence":
                if field_part not in confidence_cols:
                    raise BoppArgumentError(
                        f"Field '{field_part}' not found in confidence columns: {list(confidence_cols.keys())}."
                    )
                return confidence_cols[field_part]
            else:
                raise BoppArgumentError(
                    f"Unrecognized facet prefix '{facet_part}' in sort key '{key}'."
                )

        matching_sources: list[tuple[str, list[Any]]] = []
        if key in payload_cols:
            matching_sources.append(("payload", payload_cols[key]))
        if key in extent_cols:
            vals = extent_cols[key]
            extent_tag = _get_tag(cast(msgspec.Struct, extent))
            if (extent_tag, key) in FRACTION_AXES:
                vals = [
                    Fraction(*v) if isinstance(v, (list, tuple)) else Fraction(v)
                    for v in vals
                ]
            matching_sources.append(("extent", vals))
        if key in confidence_cols:
            matching_sources.append(("confidence", confidence_cols[key]))
        for g in groupings_list:
            if g.name == key:
                matching_sources.append(("groupings", g.ids))

        if not matching_sources:
            raise BoppArgumentError(
                f"Sort key '{key}' not found in any annotation facet (payload, extent, confidence, groupings)."
            )
        if len(matching_sources) > 1:
            sources_str = ", ".join(f"{src}.{key}" for src, _ in matching_sources)
            raise BoppArgumentError(
                f"Ambiguous sort key '{key}' found in multiple facets. Please specify explicitly: {sources_str}."
            )
        return matching_sources[0][1]

    resolved_columns = [_resolve_sort_key(k) for k in by_list]

    if n_obs <= 1:
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

    indices = list(range(n_obs))
    for col_values, asc in reversed(list(zip(resolved_columns, asc_list))):
        indices.sort(key=lambda idx: col_values[idx], reverse=not asc)

    new_payload = _subset_struct_lists(payload, indices)
    new_extent = _subset_struct_lists(extent, indices)
    new_confidence = _subset_struct_lists(confidence, indices)
    new_groupings = _subset_groupings(groupings, indices)

    return _rebuild_annotation(
        annotation,
        parents=new_parents,
        sandbox=new_sandbox,
        metadata=derived_metadata,
        payload=new_payload,  # type: ignore[arg-type]
        extent=new_extent,
        confidence=new_confidence,
        groupings=new_groupings,
    )


def shift(
    annotation: BoppBase,
    displacement: (float | Fraction | list[int] | tuple[int, int] | dict[str, Any]),
    *,
    target_field: str | None = None,
) -> BoppBase:
    """
    Offset all extents in an Annotation by a given additive displacement.

    Coordinates along the target axis or axes are shifted by adding `displacement`
    (pos -> pos + displacement). For interval and box extents, origin coordinates
    are shifted while spans (duration, width, height, length) remain unchanged.
    Displacement formats are validated according to the extent's coordinate types
    (e.g., Fraction or integer bounds for rational/discrete axes).

    Parameters
    ----------
    annotation : BoppBase
        The input Annotation instance whose extent will be shifted.
    displacement : float, int, Fraction, list[int], tuple[int, int], or dict[str, Any]
        Additive displacement to apply. Can be a numeric or rational scalar,
        or a dictionary mapping coordinate field names to displacements (e.g. {'x': 10.0, 'y': 20.0}).
    target_field : str or None, optional
        The coordinate field along which to apply a scalar displacement. If None,
        inferred from the default target field of the extent. Extents without a canonical
        default (e.g. 'pixel_box') require `target_field` or a dictionary `displacement`.
        Mutually exclusive with dictionary displacements.

    Returns
    -------
    BoppBase
        A new derived Annotation instance with shifted extent coordinates.

    Raises
    ------
    BoppArgumentError
        If the annotation has no extent, arguments are conflicting or invalid,
        or the displacement format is unsupported for the extent type.

    Examples
    --------
    Shift a time_interval extent forward by 1.5 seconds:

    >>> import bopp
    >>> ann = bopp.create(
    ...     media_id="track_1",
    ...     payload_type="tag",
    ...     extent_type="time_interval",
    ...     time=[0.0, 2.0],
    ...     duration=[1.0, 2.0],
    ...     value=["intro", "verse"],
    ... )
    >>> shifted = bopp.transforms.shift(ann, 1.5)
    >>> shifted.extent.time
    [1.5, 3.5]
    >>> shifted.extent.duration
    [1.0, 2.0]

    Shift a 2D pixel_box extent along x and y via dictionary:

    >>> ann_box = bopp.create(
    ...     media_id="image_1",
    ...     payload_type="tag",
    ...     extent_type="pixel_box",
    ...     x=[10.0],
    ...     width=[50.0],
    ...     y=[20.0],
    ...     height=[30.0],
    ...     value=["face"],
    ... )
    >>> shifted_box = bopp.transforms.shift(ann_box, {"x": 5.0, "y": -10.0})
    >>> shifted_box.extent.x
    [15.0]
    >>> shifted_box.extent.y
    [10.0]

    Shift a rational score extent by a Fraction or reduced pair:

    >>> ann_score = bopp.create(
    ...     media_id="score_1",
    ...     payload_type="tag",
    ...     extent_type="score_quarter",
    ...     quarter=[[0, 1], [1, 2]],
    ...     value=["note1", "note2"],
    ... )
    >>> shifted_score = bopp.transforms.shift(ann_score, [1, 2])
    >>> shifted_score.extent.quarter
    [[1, 2], [1, 1]]
    """
    extent = getattr(annotation, "extent", None)
    if extent is None or extent is msgspec.UNSET:
        raise BoppArgumentError("shift requires an annotation with an extent.")

    extent_tag = _get_tag(extent)
    if extent_tag is None:
        raise BoppArgumentError("Annotation extent has no valid schema tag.")

    if isinstance(displacement, dict):
        if target_field is not None:
            raise BoppArgumentError(
                "Cannot specify both 'target_field' and a dictionary 'displacement'."
            )
        if not displacement:
            raise BoppArgumentError("Dictionary 'displacement' must not be empty.")
        axis_displacements = dict(displacement)
    else:
        resolved_field = target_field
        if resolved_field is None:
            resolved_field = DEFAULT_TARGET_FIELDS.get(extent_tag)
            if resolved_field is None:
                raise BoppArgumentError(
                    f"Extent '{extent_tag}' does not have a default target field. "
                    "Please specify 'target_field' explicitly or pass a dictionary displacement."
                )
        axis_displacements = {resolved_field: displacement}

    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Shifting")

    if extent_tag == "time_uniform":
        for fld in axis_displacements:
            if fld != "time":
                raise BoppArgumentError(
                    f"Unsupported target_field '{fld}' for extent tag 'time_uniform'."
                )
        disp = axis_displacements["time"]
        if type(disp) is not int and type(disp) is not float:
            raise BoppArgumentError(
                "Displacement for 'time_uniform' must be an int or float."
            )
        if type(disp) is float and (math.isnan(disp) or math.isinf(disp)):
            raise BoppArgumentError("Displacement must be finite and not NaN.")

        disp_f = float(disp)
        new_start_time = float(extent.start_time) + disp_f
        tu_updates: dict[str, Any] = {"start_time": new_start_time}
        old_min = getattr(extent, "min_time", msgspec.UNSET)
        old_max = getattr(extent, "max_time", msgspec.UNSET)
        if old_min not in (None, msgspec.UNSET):
            tu_updates["min_time"] = float(old_min) + disp_f
        if old_max not in (None, msgspec.UNSET):
            tu_updates["max_time"] = float(old_max) + disp_f

        new_extent = msgspec.structs.replace(extent, **tu_updates)
        derived_metadata = DerivedAnnotationMetadata(
            transform="shift",
            parameters={
                "displacement": disp_f,
                "target_field": target_field or "time",
            },
        )
        return _rebuild_annotation(
            annotation,
            parents=new_parents,
            sandbox=new_sandbox,
            metadata=derived_metadata,
            extent=new_extent,
            payload=copy.deepcopy(annotation.payload),  # type: ignore[attr-defined]
            confidence=copy.deepcopy(getattr(annotation, "confidence", msgspec.UNSET)),
            groupings=copy.deepcopy(getattr(annotation, "groupings", msgspec.UNSET)),
        )

    extent_updates: dict[str, list[Any]] = {}
    scalar_updates: dict[str, Any] = {}

    for fld, disp in axis_displacements.items():
        axis_config = AXIS_CONFIGS.get((extent_tag, fld))
        if axis_config is None:
            raise BoppArgumentError(
                f"Unsupported target_field '{fld}' for extent tag '{extent_tag}'."
            )

        fraction_axis = (extent_tag, fld) in FRACTION_AXES
        integer_axis = (extent_tag, fld) in INTEGER_AXES

        kind, field_a, field_b = axis_config

        if fraction_axis:
            disp_coerced = _coerce_trim_bound(disp)
            if (
                not isinstance(disp_coerced, (int, Fraction))
                or type(disp_coerced) is bool
            ):
                raise BoppArgumentError(
                    f"Displacement for fraction axis '{fld}' must be an int, Fraction, or [num, den] pair."
                )
            disp_frac = Fraction(disp_coerced)

            if kind == "point":
                old_vals = getattr(extent, field_a)
                extent_updates[field_a] = [
                    _fraction_pair(Fraction(*v) + disp_frac) for v in old_vals
                ]
            else:  # kind == "origin_span"
                old_origins = getattr(extent, field_a)
                extent_updates[field_a] = [
                    _fraction_pair(Fraction(*v) + disp_frac) for v in old_origins
                ]

            min_attr, max_attr = _get_extent_axis_bound_attrs(extent_tag, fld)
            old_min = getattr(extent, min_attr, msgspec.UNSET)
            old_max = getattr(extent, max_attr, msgspec.UNSET)
            if old_min not in (None, msgspec.UNSET):
                scalar_updates[min_attr] = _fraction_pair(
                    _to_coord(old_min) + disp_frac
                )
            if old_max not in (None, msgspec.UNSET):
                scalar_updates[max_attr] = _fraction_pair(
                    _to_coord(old_max) + disp_frac
                )

        elif integer_axis:
            if type(disp) is not int:
                raise BoppArgumentError(
                    f"Displacement for integer axis '{fld}' must be an int, got {type(disp).__name__}."
                )
            disp_int = disp

            if kind == "point":
                old_vals = getattr(extent, field_a)
                extent_updates[field_a] = [round(v + disp_int) for v in old_vals]
            else:  # kind == "origin_span"
                old_origins = getattr(extent, field_a)
                extent_updates[field_a] = [round(v + disp_int) for v in old_origins]

            min_attr, max_attr = _get_extent_axis_bound_attrs(extent_tag, fld)
            old_min = getattr(extent, min_attr, msgspec.UNSET)
            old_max = getattr(extent, max_attr, msgspec.UNSET)
            if old_min not in (None, msgspec.UNSET):
                scalar_updates[min_attr] = round(_to_coord(old_min) + disp_int)
            if old_max not in (None, msgspec.UNSET):
                scalar_updates[max_attr] = round(_to_coord(old_max) + disp_int)

        else:
            if type(disp) is not int and type(disp) is not float:
                raise BoppArgumentError(
                    f"Displacement for axis '{fld}' must be an int or float, got {type(disp).__name__}."
                )
            if type(disp) is float and (math.isnan(disp) or math.isinf(disp)):
                raise BoppArgumentError("Displacement must be finite and not NaN.")
            disp_num = float(disp)

            if kind == "point":
                old_vals = getattr(extent, field_a)
                extent_updates[field_a] = [float(v + disp_num) for v in old_vals]
            elif kind == "origin_span":
                old_origins = getattr(extent, field_a)
                extent_updates[field_a] = [float(v + disp_num) for v in old_origins]
            else:  # kind == "min_max"
                field_b_str = cast(str, field_b)
                old_mins = getattr(extent, field_a)
                old_maxs = getattr(extent, field_b_str)
                extent_updates[field_a] = [float(v + disp_num) for v in old_mins]
                extent_updates[field_b_str] = [float(v + disp_num) for v in old_maxs]

            min_attr, max_attr = _get_extent_axis_bound_attrs(extent_tag, fld)
            old_min = getattr(extent, min_attr, msgspec.UNSET)
            old_max = getattr(extent, max_attr, msgspec.UNSET)
            if old_min not in (None, msgspec.UNSET):
                scalar_updates[min_attr] = float(_to_coord(old_min) + disp_num)
            if old_max not in (None, msgspec.UNSET):
                scalar_updates[max_attr] = float(_to_coord(old_max) + disp_num)

    all_extent_updates = {**extent_updates, **scalar_updates}
    new_extent = msgspec.structs.replace(extent, **all_extent_updates)

    serializable_disp = (
        {k: _trim_bound_parameter(v) for k, v in axis_displacements.items()}
        if isinstance(displacement, dict)
        else _trim_bound_parameter(displacement)
    )
    derived_metadata = DerivedAnnotationMetadata(
        transform="shift",
        parameters={
            "displacement": serializable_disp,
            "target_field": (
                resolved_field if not isinstance(displacement, dict) else None
            ),
        },
    )

    return _rebuild_annotation(
        annotation,
        parents=new_parents,
        sandbox=new_sandbox,
        metadata=derived_metadata,
        extent=new_extent,
        payload=copy.deepcopy(annotation.payload),  # type: ignore[attr-defined]
        confidence=copy.deepcopy(getattr(annotation, "confidence", msgspec.UNSET)),
        groupings=copy.deepcopy(getattr(annotation, "groupings", msgspec.UNSET)),
    )


def union(
    annotation: BoppBase | Sequence[BoppBase],
    *annotations: BoppBase,
) -> BoppBase:
    """
    Combine two or more compatible annotations into a single derived annotation.

    All operands must share the same `media_id`, `bopp_version`, and facet type
    discriminators across payload, extent, and confidence facets. Grouping layer counts
    and names must also match. Columnar array fields are concatenated across operands,
    while top-level structural fields (`sandbox`, `metadata`) are propagated from the
    first given operand. Derivation lineage records the IDs of all input operands in `parents`.

    Parameters
    ----------
    annotation : BoppBase or Sequence of BoppBase
        The first annotation instance, or a sequence of annotations to union.
    *annotations : BoppBase
        Additional annotation instances to combine.

    Returns
    -------
    BoppBase
        A new derived Annotation instance containing the concatenated observations.

    Raises
    ------
    BoppArgumentError
        If fewer than two annotations are provided, or operands have mismatched
        facet types, media IDs, bopp versions, or grouping layers. Also raises if
        operands use 'time_uniform' extents.

    Examples
    --------
    Combine two time_interval annotations from the same media:

    >>> import bopp
    >>> ann1 = bopp.create(
    ...     media_id="track_1",
    ...     payload_type="tag",
    ...     extent_type="time_interval",
    ...     time=[0.0, 2.0],
    ...     duration=[2.0, 2.0],
    ...     value=["intro", "verse"],
    ... )
    >>> ann2 = bopp.create(
    ...     media_id="track_1",
    ...     payload_type="tag",
    ...     extent_type="time_interval",
    ...     time=[4.0],
    ...     duration=[3.0],
    ...     value=["chorus"],
    ... )
    >>> combined = bopp.transforms.union(ann1, ann2)
    >>> combined.extent.time
    [0.0, 2.0, 4.0]
    >>> combined.payload.value
    ['intro', 'verse', 'chorus']
    """
    anns: list[Any]
    if isinstance(annotation, Sequence) and not isinstance(annotation, BoppBase):
        anns = list(annotation) + list(annotations)
    else:
        anns = [annotation] + list(annotations)

    if len(anns) < 2:
        raise BoppArgumentError(
            f"union requires at least 2 annotations, got {len(anns)}."
        )

    for idx, a in enumerate(anns):
        if not isinstance(a, BoppBase):
            raise BoppArgumentError(
                f"Operand at index {idx} is not an Annotation instance (got {type(a).__name__})."
            )

    first = anns[0]
    first_media_id = getattr(first, "media_id", None)
    first_version = getattr(first, "bopp_version", None)

    for idx, a in enumerate(anns[1:], start=1):
        a_mid = getattr(a, "media_id", None)
        if a_mid != first_media_id:
            raise BoppArgumentError(
                f"Annotation at index {idx} has media_id '{a_mid}', matching '{first_media_id}' required."
            )
        a_ver = getattr(a, "bopp_version", None)
        if a_ver != first_version:
            raise BoppArgumentError(
                f"Annotation at index {idx} has bopp_version '{a_ver}', matching '{first_version}' required."
            )

    first_payload = getattr(first, "payload", None)
    first_p_tag = _get_tag(first_payload) if first_payload is not None else None
    if first_p_tag is None:
        raise BoppArgumentError("First annotation has no valid payload.")
    for idx, a in enumerate(anns[1:], start=1):
        a_p = getattr(a, "payload", None)
        a_p_tag = _get_tag(a_p) if a_p is not None else None
        if a_p_tag != first_p_tag:
            raise BoppArgumentError(
                f"Payload type mismatch at index {idx}: '{a_p_tag}' != '{first_p_tag}'."
            )

    first_extent = getattr(first, "extent", None)
    first_e_tag = (
        _get_tag(first_extent) if first_extent not in (None, msgspec.UNSET) else None
    )
    if first_e_tag == "time_uniform":
        raise BoppArgumentError(
            "union does not support 'time_uniform' extents because combining multiple segments "
            "produces non-uniform rasters. Convert with bopp.to_times() first."
        )
    for idx, a in enumerate(anns[1:], start=1):
        a_e = getattr(a, "extent", None)
        a_e_tag = _get_tag(a_e) if a_e not in (None, msgspec.UNSET) else None
        if a_e_tag == "time_uniform":
            raise BoppArgumentError(
                f"Annotation at index {idx} has 'time_uniform' extent, which is unsupported in union."
            )
        if a_e_tag != first_e_tag:
            raise BoppArgumentError(
                f"Extent type mismatch at index {idx}: '{a_e_tag}' != '{first_e_tag}'."
            )

    first_conf = getattr(first, "confidence", None)
    first_c_tag = (
        _get_tag(first_conf) if first_conf not in (None, msgspec.UNSET) else None
    )
    for idx, a in enumerate(anns[1:], start=1):
        a_c = getattr(a, "confidence", None)
        a_c_tag = _get_tag(a_c) if a_c not in (None, msgspec.UNSET) else None
        if a_c_tag != first_c_tag:
            raise BoppArgumentError(
                f"Confidence type mismatch at index {idx}: '{a_c_tag}' != '{first_c_tag}'."
            )

    first_groupings = getattr(first, "groupings", None)
    first_has_g = first_groupings not in (None, msgspec.UNSET)
    for idx, a in enumerate(anns[1:], start=1):
        a_g = getattr(a, "groupings", None)
        a_has_g = a_g not in (None, msgspec.UNSET)
        if first_has_g != a_has_g:
            raise BoppArgumentError(
                f"Groupings mismatch: annotation at index {idx} has_groupings={a_has_g}, but first operand has_groupings={first_has_g}."
            )
        if first_has_g:
            first_g_list = cast(list[Any], first_groupings)
            a_g_list = cast(list[Any], a_g)
            if len(first_g_list) != len(a_g_list):
                raise BoppArgumentError(
                    f"Groupings count mismatch at index {idx}: {len(a_g_list)} layers != {len(first_g_list)} layers."
                )
            for layer_idx, (g1, g2) in enumerate(zip(first_g_list, a_g_list)):
                if g1.name != g2.name:
                    raise BoppArgumentError(
                        f"Grouping layer name mismatch at index {idx}, layer {layer_idx}: '{g2.name}' != '{g1.name}'."
                    )

    merged_groupings: list[Any] = []
    if first_has_g:
        first_g_list = cast(list[Any], first_groupings)
        for layer_idx in range(len(first_g_list)):
            combined_ids = [
                x for a in anns for x in getattr(a, "groupings", [])[layer_idx].ids
            ]
            merged_labels: dict[str, int] = {}
            for a in anns:
                a_groupings = getattr(a, "groupings", [])
                lbls = getattr(a_groupings[layer_idx], "labels", msgspec.UNSET)
                if lbls not in (None, msgspec.UNSET):
                    for label_str, gid in lbls.items():
                        if (
                            label_str in merged_labels
                            and merged_labels[label_str] != gid
                        ):
                            raise BoppArgumentError(
                                f"Conflicting label mapping in grouping layer '{first_g_list[layer_idx].name}': "
                                f"label '{label_str}' maps to both {merged_labels[label_str]} and {gid}."
                            )
                        merged_labels[label_str] = gid

            id_to_label: dict[int, str] = {}
            for l_str, gid in merged_labels.items():
                if gid in id_to_label and id_to_label[gid] != l_str:
                    raise BoppArgumentError(
                        f"Conflicting label mapping in grouping layer '{first_g_list[layer_idx].name}': "
                        f"ID {gid} is assigned to both '{id_to_label[gid]}' and '{l_str}'."
                    )
                id_to_label[gid] = l_str

            lbl_val = merged_labels if merged_labels else msgspec.UNSET
            g_candidate = msgspec.structs.replace(
                first_g_list[layer_idx],
                ids=combined_ids,
                labels=lbl_val,
                disjoint=False,
                exhaustive=False,
            )
            merged_groupings.append(g_candidate)

    first_payload_struct = cast(msgspec.Struct, first_payload)
    p_updates: dict[str, list[Any]] = {}
    payload_fields = _get_facet_list_fields(first_payload_struct)
    for idx, a in enumerate(anns[1:], start=1):
        incoming_fields = _get_facet_list_fields(getattr(a, "payload", None))
        if incoming_fields.keys() != payload_fields.keys():
            raise BoppArgumentError(
                f"Payload column mismatch at index {idx}: optional columns must have matching presence."
            )
    for fname in payload_fields:
        p_updates[fname] = [
            x for a in anns for x in getattr(getattr(a, "payload", None), fname)
        ]
    for a in anns[1:]:
        a_payload = getattr(a, "payload", None)
        for fname in first_payload_struct.__struct_fields__:
            original = getattr(first_payload_struct, fname)
            incoming = getattr(a_payload, fname)
            if (
                not isinstance(original, list)
                and not isinstance(incoming, list)
                and original != incoming
            ):
                raise BoppArgumentError(
                    f"Scalar payload field '{fname}' must match across union operands."
                )
    new_payload: msgspec.Struct = msgspec.structs.replace(
        first_payload_struct, **p_updates
    )

    new_confidence: msgspec.Struct | msgspec.UnsetType = msgspec.UNSET
    if first_c_tag is not None:
        first_conf_struct = cast(msgspec.Struct, first_conf)
        c_updates: dict[str, list[Any]] = {}
        confidence_fields = _get_facet_list_fields(first_conf_struct)
        for idx, a in enumerate(anns[1:], start=1):
            incoming_fields = _get_facet_list_fields(getattr(a, "confidence", None))
            if incoming_fields.keys() != confidence_fields.keys():
                raise BoppArgumentError(
                    f"Confidence column mismatch at index {idx}: optional columns must have matching presence."
                )
        for fname in confidence_fields:
            c_updates[fname] = [
                x for a in anns for x in getattr(getattr(a, "confidence", None), fname)
            ]
        for a in anns[1:]:
            a_conf = getattr(a, "confidence", None)
            for fname in first_conf_struct.__struct_fields__:
                original = getattr(first_conf_struct, fname)
                incoming = getattr(a_conf, fname)
                if (
                    not isinstance(original, list)
                    and not isinstance(incoming, list)
                    and original != incoming
                ):
                    raise BoppArgumentError(
                        f"Scalar confidence field '{fname}' must match across union operands."
                    )
        new_confidence = msgspec.structs.replace(first_conf_struct, **c_updates)

    new_extent: msgspec.Struct | msgspec.UnsetType = msgspec.UNSET
    if first_e_tag is not None:
        first_extent_struct = cast(msgspec.Struct, first_extent)
        e_updates: dict[str, Any] = {}
        for fname in _get_facet_list_fields(first_extent_struct):
            e_updates[fname] = [
                x for a in anns for x in getattr(getattr(a, "extent", None), fname)
            ]

        specs = EXTENT_AXES_METADATA.get(first_e_tag, [])
        is_fraction = first_e_tag in FRACTION_EXTENT_TAGS
        for spec in specs:
            min_attr = spec["min_attr"]
            max_attr = spec["max_attr"]
            dim = spec["dim"]
            has_explicit_min = any(
                getattr(getattr(a, "extent", None), min_attr, msgspec.UNSET)
                not in (None, msgspec.UNSET)
                for a in anns
            )
            has_explicit_max = any(
                getattr(getattr(a, "extent", None), max_attr, msgspec.UNSET)
                not in (None, msgspec.UNSET)
                for a in anns
            )
            if has_explicit_min:
                eff_mins = [
                    get_extent_bounds(getattr(a, "extent", None)).get(
                        dim, (None, None)
                    )[0]
                    for a in anns
                    if getattr(a, "extent", None) not in (None, msgspec.UNSET)
                ]
                min_val = min(_to_coord(v) for v in eff_mins if v is not None)
                e_updates[min_attr] = (
                    _fraction_pair(min_val) if is_fraction else min_val
                )
            if has_explicit_max:
                eff_maxs = [
                    get_extent_bounds(getattr(a, "extent", None)).get(
                        dim, (None, None)
                    )[1]
                    for a in anns
                    if getattr(a, "extent", None) not in (None, msgspec.UNSET)
                ]
                max_val = max(_to_coord(v) for v in eff_maxs if v is not None)
                e_updates[max_attr] = (
                    _fraction_pair(max_val) if is_fraction else max_val
                )

        candidate_extent = msgspec.structs.replace(
            first_extent_struct, disjoint=False, exhaustive=False, **e_updates
        )
        total_n = len(next(iter(e_updates.values()), []))
        new_disjoint = is_disjoint(candidate_extent)
        new_exhaustive = is_exhaustive(candidate_extent, N=total_n)
        new_extent = msgspec.structs.replace(
            candidate_extent, disjoint=new_disjoint, exhaustive=new_exhaustive
        )

    if merged_groupings:
        total_obs = len(next(iter(p_updates.values()), []))
        final_groupings = []
        for g in merged_groupings:
            g_disj = new_extent is not msgspec.UNSET and is_disjoint(
                new_extent, grouping=g
            )
            g_exh = is_exhaustive(
                new_extent if new_extent is not msgspec.UNSET else None,
                grouping=g,
                N=total_obs,
            )
            final_groupings.append(
                msgspec.structs.replace(g, disjoint=g_disj, exhaustive=g_exh)
            )
        merged_groupings = final_groupings

    all_parents: list[str] = []
    for a in anns:
        a_parents = getattr(a, "parents", None)
        if a_parents not in (None, msgspec.UNSET):
            for p in a_parents:
                if p not in all_parents:
                    all_parents.append(p)
        a_id = getattr(a, "id", None)
        if a_id in (None, msgspec.UNSET):
            warnings.warn(
                "Unioning an annotation with no ID. The resulting annotation will not include it in parent lineage.",
                UserWarning,
            )
        elif a_id not in all_parents:
            all_parents.append(a_id)

    new_sandbox = copy.deepcopy(getattr(first, "sandbox", msgspec.UNSET))
    new_metadata = copy.deepcopy(getattr(first, "metadata", msgspec.UNSET))

    return _rebuild_annotation(
        first,
        parents=all_parents,
        sandbox=new_sandbox,
        metadata=new_metadata,
        payload=new_payload,
        extent=new_extent,
        confidence=new_confidence,
        groupings=merged_groupings if first_has_g else msgspec.UNSET,
    )


def to_uniform_times(
    annotation: BoppBase,
    *,
    sampling_period: float,
    start_time: float | None = None,
    end_time: float | None = None,
    method: Literal["previous", "nearest"] = "previous",
    overlap: Literal["latest", "first"] = "latest",
    fill_value: Any = None,
) -> BoppBase:
    """
    Convert a disjoint temporal Annotation to the 'time_uniform' regular raster extent.

    The input annotation must have a temporal extent and must be strictly disjoint
    across observations (grouping-level disjointedness alone is not sufficient).
    Timesteps are sampled regularly with period `sampling_period`.

    Parameters
    ----------
    annotation : BoppBase
        The input Annotation instance to convert.
    sampling_period : float
        Uniform sampling period (timestep) in seconds (> 0).
    start_time : float or None, optional
        Start time of the uniform raster in seconds. If None, inferred from the earliest observation.
    end_time : float or None, optional
        End time bound of the uniform raster in seconds. If None, inferred from the latest observation.
    method : {"previous", "nearest"}, default "previous"
        Interpolation method when sampling point extents ('time').
    overlap : {"latest", "first"}, default "latest"
        Disambiguation strategy when observation boundaries touch:
        - "latest": Pick the interval starting latest.
        - "first": Pick the interval starting earliest.
    fill_value : Any or dict of str to Any, optional
        Value to fill when a sample timestamp falls in a gap. Default is None.

    Returns
    -------
    BoppBase
        A new derived Annotation instance with extent type `time_uniform` (`UniformTimeExtent`).

    Raises
    ------
    BoppArgumentError
        If arguments are invalid, the input extent is missing or incompatible, or
        the input extent is not disjoint.

    Examples
    --------
    Convert a disjoint time_interval annotation to a uniform time raster:

    >>> import bopp
    >>> ann = bopp.create(
    ...     media_id="track_1",
    ...     payload_type="tag",
    ...     extent_type="time_interval",
    ...     time=[0.0, 2.0],
    ...     duration=[2.0, 2.0],
    ...     value=["verse", "chorus"],
    ... )
    >>> tu = bopp.transforms.to_uniform_times(ann, sampling_period=1.0)
    >>> tu.extent.start_time
    0.0
    >>> tu.extent.sampling_period
    1.0
    >>> tu.payload.value
    ['verse', 'verse', 'chorus', 'chorus', 'chorus']
    """
    if type(sampling_period) not in (int, float) or type(sampling_period) is bool:
        raise BoppArgumentError(
            f"sampling_period must be a positive number, got {type(sampling_period).__name__}."
        )
    if (
        sampling_period <= 0
        or math.isnan(sampling_period)
        or math.isinf(sampling_period)
    ):
        raise BoppArgumentError(
            f"sampling_period must be positive (> 0) and finite, got {sampling_period}."
        )

    if method not in ("previous", "nearest"):
        raise BoppArgumentError(
            f"Invalid method '{method}'. Must be 'previous' or 'nearest'."
        )
    if overlap not in ("latest", "first"):
        raise BoppArgumentError(
            f"Invalid overlap strategy '{overlap}'. Must be 'latest' or 'first'."
        )

    extent = getattr(annotation, "extent", None)
    if extent is None or extent is msgspec.UNSET:
        raise BoppArgumentError(
            "to_uniform_times requires an annotation with a temporal extent."
        )

    if not is_disjoint(extent):
        raise BoppArgumentError(
            "to_uniform_times requires the input extent to be disjoint. "
            "Overlapping observations cannot be represented on a uniform raster grid."
        )

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
            f"Extent type '{extent_tag}' is incompatible with to_uniform_times. "
            f"Supported extents are {valid_temporal_extents}."
        )

    payload: msgspec.Struct = annotation.payload  # type: ignore[attr-defined]
    confidence = getattr(annotation, "confidence", None)
    groupings = getattr(annotation, "groupings", None)

    payload_cols = _get_facet_list_fields(payload)
    confidence_cols = _get_facet_list_fields(confidence)

    has_column_facet = (
        bool(payload_cols)
        or bool(confidence_cols)
        or (groupings not in (None, msgspec.UNSET) and bool(groupings))
    )
    if not has_column_facet:
        raise BoppArgumentError(
            "to_uniform_times requires at least one columnar field in payload, "
            "confidence, or groupings to preserve observation counts under UniformTimeExtent."
        )

    if extent_tag != "time_uniform":
        n_source_obs = len(next(iter(_get_facet_list_fields(extent).values()), []))
    else:
        source_cols = payload_cols or confidence_cols
        n_source_obs = len(next(iter(source_cols.values()), []))
        if not source_cols and groupings not in (None, msgspec.UNSET):
            n_source_obs = len(groupings[0].ids) if groupings else 0

    if n_source_obs == 0 and (start_time is None or end_time is None):
        raise BoppArgumentError(
            "Cannot infer temporal bounds from empty annotation. Specify 'start_time' and 'end_time'."
        )

    if n_source_obs > 0:
        if extent_tag in ("time", "spherical_time", "cartesian_time"):
            time_arr = extent.time  # type: ignore[union-attr]
            inferred_min = float(min(time_arr))
            inferred_max = float(max(time_arr))
        elif extent_tag == "time_uniform":
            st_val = float(extent.start_time)  # type: ignore[union-attr]
            sp_val = float(extent.sampling_period)  # type: ignore[union-attr]
            inferred_min = st_val
            inferred_max = st_val + max(0, n_source_obs - 1) * sp_val
        else:
            time_arr = extent.time  # type: ignore[union-attr]
            dur_arr = extent.duration  # type: ignore[union-attr]
            inferred_min = float(min(time_arr))
            inferred_max = float(max(t + d for t, d in zip(time_arr, dur_arr)))
    else:
        inferred_min = 0.0
        inferred_max = 0.0

    actual_start = inferred_min if start_time is None else float(start_time)
    actual_end = inferred_max if end_time is None else float(end_time)

    if not (math.isfinite(actual_start) and math.isfinite(actual_end)):
        raise BoppArgumentError(
            f"start_time ({actual_start}) and end_time ({actual_end}) must be finite numbers."
        )

    if actual_start > actual_end:
        raise BoppArgumentError(
            f"start_time ({actual_start}) must be <= end_time ({actual_end})."
        )

    payload_fill_map: dict[str, Any] = {}
    confidence_fill_map: dict[str, Any] = {}
    grouping_fill_map: dict[str, int] = {}

    if isinstance(fill_value, dict):
        for col_name in payload_cols:
            if col_name in fill_value:
                payload_fill_map[col_name] = fill_value[col_name]
            else:
                raise BoppArgumentError(
                    f"Missing fill value for payload field '{col_name}' in fill_value dictionary."
                )
        for col_name in confidence_cols:
            if col_name in fill_value:
                confidence_fill_map[col_name] = fill_value[col_name]
            else:
                confidence_fill_map[col_name] = 0.0
        if groupings not in (None, msgspec.UNSET):
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

    step = float(sampling_period)
    n_steps = math.floor((actual_end - actual_start) / step + 1e-9) + 1
    target_times = [actual_start + i * step for i in range(n_steps)]

    matched_indices: list[int | None] = []

    if n_source_obs == 0:
        matched_indices = [None] * len(target_times)
    elif extent_tag in (
        "time_interval",
        "time_frequency_box",
        "spherical_time_interval",
        "cartesian_time_interval",
    ):
        t_starts = extent.time  # type: ignore[union-attr]
        durations = extent.duration  # type: ignore[union-attr]
        for t in target_times:
            covers = [
                i
                for i, (ts, dur) in enumerate(zip(t_starts, durations))
                if ts - 1e-9 <= t <= ts + dur + 1e-9
            ]
            if not covers:
                matched_indices.append(None)
            elif overlap == "first":
                matched_indices.append(min(covers, key=lambda i: t_starts[i]))
            else:  # overlap == "latest"
                matched_indices.append(max(covers, key=lambda i: t_starts[i]))
    else:
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
                    matched_indices.append(None)
                else:
                    matched_indices.append(
                        max(prev_candidates, key=lambda i: src_times[i])
                    )
            else:  # method == "nearest"
                best_idx = min(
                    range(len(src_times)), key=lambda i: abs(src_times[i] - t)
                )
                matched_indices.append(best_idx)

    if None in matched_indices:
        _validate_fill_values(payload, payload_fill_map, "payload")
        if confidence not in (None, msgspec.UNSET):
            _validate_fill_values(confidence, confidence_fill_map, "confidence")

    new_payload = _resample_struct(
        payload, payload_cols, matched_indices, payload_fill_map
    )
    new_confidence = _resample_struct(
        confidence, confidence_cols, matched_indices, confidence_fill_map
    )
    new_groupings = _resample_groupings(groupings, matched_indices, grouping_fill_map)

    candidate_extent = UniformTimeExtent(
        start_time=float(actual_start),
        sampling_period=float(step),
        disjoint=True,
        exhaustive=False,
        min_time=float(actual_start),
        max_time=float(actual_start + max(0, n_steps - 1) * step),
    )
    tu_exh = bool(n_steps > 0 and is_exhaustive(candidate_extent, N=n_steps))
    new_extent = msgspec.structs.replace(candidate_extent, exhaustive=tu_exh)

    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Converting")
    derived_metadata = DerivedAnnotationMetadata(
        transform="to_uniform_times",
        parameters={
            "sampling_period": step,
            "start_time": actual_start,
            "end_time": actual_end,
            "method": method,
            "overlap": overlap,
        },
    )

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


def to_pixels(
    annotation: BoppBase,
    *,
    pixels: Sequence[tuple[float, float]] | None = None,
    x: Sequence[float] | None = None,
    y: Sequence[float] | None = None,
    overlap: Literal["latest", "first", "multiple"] = "latest",
    fill_value: Any = None,
) -> BoppBase:
    """
    Sample observation data from a 'pixel_box' (or 'pixel') Annotation at discrete pixel coordinates.

    Evaluates observations at the requested 2D pixel coordinates (x, y). For 'pixel_box' extents,
    observations covering each coordinate (x_origin <= x <= x_origin + width,
    y_origin <= y <= y_origin + height) are sampled. For point 'pixel' extents, exact coordinate
    matches are sampled. Payload, confidence, and groupings are resampled in parallel.

    Parameters
    ----------
    annotation : BoppBase
        The input Annotation instance to sample.
    pixels : Sequence of tuple of (float, float) or None, optional
        Explicit sequence of (x, y) coordinates at which to sample observations.
        Mutually exclusive with `x` and `y`.
    x : Sequence of float or None, optional
        Array of x-coordinates. Must be paired with `y`.
    y : Sequence of float or None, optional
        Array of y-coordinates. Must be paired with `x`.
    overlap : {"latest", "first", "multiple"}, default "latest"
        Disambiguation strategy when multiple pixel boxes cover a sample coordinate:
        - "latest": Pick the box appearing latest in the annotation.
        - "first": Pick the box appearing earliest in the annotation.
        - "multiple": Emit duplicate sample coordinates, one for each covering box.
    fill_value : Any or dict of str to Any, optional
        Value to fill when a sample coordinate falls in a gap. Default is None.

    Returns
    -------
    BoppBase
        A new derived Annotation instance with extent type `pixel` (`Pixel` struct).

    Raises
    ------
    BoppArgumentError
        If arguments are invalid or the extent is incompatible with pixel sampling.

    Examples
    --------
    Sample observation values from pixel boxes at explicit coordinates:

    >>> import bopp
    >>> ann = bopp.create(
    ...     media_id="frame_1",
    ...     payload_type="tag",
    ...     extent_type="pixel_box",
    ...     x=[0.0, 100.0],
    ...     width=[50.0, 50.0],
    ...     y=[0.0, 100.0],
    ...     height=[50.0, 50.0],
    ...     value=["background", "foreground"],
    ... )
    >>> sampled = bopp.transforms.to_pixels(ann, pixels=[(25.0, 25.0), (120.0, 120.0)])
    >>> sampled.extent.x
    [25.0, 120.0]
    >>> sampled.extent.y
    [25.0, 120.0]
    >>> sampled.payload.value
    ['background', 'foreground']
    """
    if pixels is not None:
        if x is not None or y is not None:
            raise BoppArgumentError("Cannot specify both 'pixels' and ('x' or 'y').")
        target_x = [float(p[0]) for p in pixels]
        target_y = [float(p[1]) for p in pixels]
    elif x is not None and y is not None:
        if len(x) != len(y):
            raise BoppArgumentError(
                f"Lengths of 'x' ({len(x)}) and 'y' ({len(y)}) must match."
            )
        target_x = [float(val) for val in x]
        target_y = [float(val) for val in y]
    else:
        raise BoppArgumentError(
            "Must specify either 'pixels' as a sequence of (x, y) tuples, or both 'x' and 'y' sequences."
        )

    if overlap not in ("latest", "first", "multiple"):
        raise BoppArgumentError(
            f"Invalid overlap strategy '{overlap}'. Must be 'latest', 'first', or 'multiple'."
        )

    extent = getattr(annotation, "extent", None)
    extent_tag = _get_tag(extent) if extent not in (None, msgspec.UNSET) else None
    if extent_tag not in ("pixel_box", "pixel", None):
        raise BoppArgumentError(
            f"Extent type '{extent_tag}' is incompatible with to_pixels. "
            "Supported extents are 'pixel_box', 'pixel', or None (global)."
        )

    payload: msgspec.Struct = annotation.payload  # type: ignore[attr-defined]
    confidence = getattr(annotation, "confidence", None)
    groupings = getattr(annotation, "groupings", None)

    payload_cols = _get_facet_list_fields(payload)
    confidence_cols = _get_facet_list_fields(confidence)

    if extent_tag in ("pixel_box", "pixel"):
        n_source_obs = len(extent.x)  # type: ignore[union-attr]
    else:
        source_cols = payload_cols or confidence_cols
        n_source_obs = len(next(iter(source_cols.values()), []))
        if not source_cols and groupings not in (None, msgspec.UNSET):
            n_source_obs = len(groupings[0].ids) if groupings else 0

    payload_fill_map: dict[str, Any] = {}
    confidence_fill_map: dict[str, Any] = {}
    grouping_fill_map: dict[str, int] = {}

    if isinstance(fill_value, dict):
        for col_name in payload_cols:
            if col_name in fill_value:
                payload_fill_map[col_name] = fill_value[col_name]
            else:
                raise BoppArgumentError(
                    f"Missing fill value for payload field '{col_name}' in fill_value dictionary."
                )
        for col_name in confidence_cols:
            if col_name in fill_value:
                confidence_fill_map[col_name] = fill_value[col_name]
            else:
                confidence_fill_map[col_name] = 0.0
        if groupings not in (None, msgspec.UNSET):
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

    out_x: list[float] = []
    out_y: list[float] = []
    matched_indices: list[int | None] = []

    if not target_x:
        pass
    elif n_source_obs == 0:
        out_x = list(target_x)
        out_y = list(target_y)
        matched_indices = [None] * len(target_x)
    elif extent_tag is None:
        all_indices = list(range(n_source_obs))
        for qx, qy in zip(target_x, target_y):
            if overlap == "multiple":
                for idx in all_indices:
                    out_x.append(qx)
                    out_y.append(qy)
                    matched_indices.append(idx)
            elif overlap == "first":
                out_x.append(qx)
                out_y.append(qy)
                matched_indices.append(all_indices[0])
            else:  # overlap == "latest"
                out_x.append(qx)
                out_y.append(qy)
                matched_indices.append(all_indices[-1])
    elif extent_tag == "pixel_box":
        bx_list = extent.x  # type: ignore[union-attr]
        bw_list = extent.width  # type: ignore[union-attr]
        by_list = extent.y  # type: ignore[union-attr]
        bh_list = extent.height  # type: ignore[union-attr]
        for qx, qy in zip(target_x, target_y):
            covers = [
                i
                for i, (bx, bw, by, bh) in enumerate(
                    zip(bx_list, bw_list, by_list, bh_list)
                )
                if bx - 1e-9 <= qx <= bx + bw + 1e-9
                and by - 1e-9 <= qy <= by + bh + 1e-9
            ]
            if not covers:
                out_x.append(qx)
                out_y.append(qy)
                matched_indices.append(None)
            elif overlap == "multiple":
                for idx in covers:
                    out_x.append(qx)
                    out_y.append(qy)
                    matched_indices.append(idx)
            elif overlap == "first":
                out_x.append(qx)
                out_y.append(qy)
                matched_indices.append(covers[0])
            else:  # overlap == "latest"
                out_x.append(qx)
                out_y.append(qy)
                matched_indices.append(covers[-1])
    else:  # extent_tag == "pixel"
        px_list = extent.x  # type: ignore[union-attr]
        py_list = extent.y  # type: ignore[union-attr]
        for qx, qy in zip(target_x, target_y):
            matches = [
                i
                for i, (px, py) in enumerate(zip(px_list, py_list))
                if px == qx and py == qy
            ]
            if not matches:
                out_x.append(qx)
                out_y.append(qy)
                matched_indices.append(None)
            elif overlap == "multiple":
                for idx in matches:
                    out_x.append(qx)
                    out_y.append(qy)
                    matched_indices.append(idx)
            elif overlap == "first":
                out_x.append(qx)
                out_y.append(qy)
                matched_indices.append(matches[0])
            else:  # overlap == "latest"
                out_x.append(qx)
                out_y.append(qy)
                matched_indices.append(matches[-1])

    if None in matched_indices:
        _validate_fill_values(payload, payload_fill_map, "payload")
        if confidence not in (None, msgspec.UNSET):
            _validate_fill_values(confidence, confidence_fill_map, "confidence")

    new_payload = _resample_struct(
        payload, payload_cols, matched_indices, payload_fill_map
    )
    new_confidence = _resample_struct(
        confidence, confidence_cols, matched_indices, confidence_fill_map
    )
    new_groupings = _resample_groupings(groupings, matched_indices, grouping_fill_map)

    is_disj = len(set(zip(out_x, out_y))) == len(out_x)
    new_extent = Pixel(x=out_x, y=out_y, disjoint=is_disj, exhaustive=False)

    if new_groupings not in (None, msgspec.UNSET):
        new_groupings = [
            msgspec.structs.replace(g, disjoint=is_disjoint(new_extent, grouping=g))
            for g in new_groupings
        ]

    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Converting")
    derived_metadata = DerivedAnnotationMetadata(
        transform="to_pixels",
        parameters={"overlap": overlap},
    )

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
