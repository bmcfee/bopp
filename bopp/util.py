from __future__ import annotations

import functools
import math
import types
import typing
from collections import defaultdict
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, get_args, get_origin

import msgspec
import polars as pl
import pyarrow as pa

from ._arrow import arrow_dec_hook, unwrap_arrow_array
from ._version import get_current_schema_version
from .base import BoppBase
from .exceptions import (
    BoppArgumentError,
    BoppIOError,
    BoppRegistryError,
    BoppValidationError,
)
from .registries import FACET_REGISTRY_KEYS, get_registry

if TYPE_CHECKING:
    import pandas as pd

# ==========================================
# 1. Struct <-> DataFrame Translators
# ==========================================


class AnnotationDataFrame(NamedTuple):
    """
    Container pairing an annotation DataFrame with its scalar metadata header.

    Attributes
    ----------
    df : pandas.DataFrame or polars.DataFrame
        DataFrame holding parallel columnar arrays from the annotation facets.
    header : dict of str to Any
        Dictionary of scalar/singleton fields and metadata.
    """

    df: pd.DataFrame | pl.DataFrame
    header: dict[str, Any]


def _get_tag(struct: msgspec.Struct) -> str | None:
    """
    Retrieve the tag value from a msgspec Struct, if defined.

    Parameters
    ----------
    struct : msgspec.Struct
        The struct from which to extract the tag.

    Returns
    -------
    str or None
        The tag string if defined, otherwise None.

    See Also
    --------
    to_dataframe : Converts an Annotation instance into a DataFrame using tag names.

    Examples
    --------
    >>> from bopp.models.v1.payload.beat import BeatPositionPayload
    >>> payload = BeatPositionPayload(value=[1.0, 2.0])
    >>> _get_tag(payload)
    'beat'
    """
    config = getattr(struct, "__struct_config__", None)
    if config is not None:
        return getattr(config, "tag", None)
    return None


@functools.cache
def _get_struct_fields(
    cls: type[msgspec.Struct],
) -> tuple[msgspec.structs.FieldInfo, ...]:
    """Return cached msgspec struct field metadata for a Struct class."""
    return msgspec.structs.fields(cls)


def _validate_dataframe_columns(
    columns: list[str], registry_or_version: dict[str, Any] | str
) -> tuple[dict[str, list[tuple[str, str, str]]], list[tuple[int, str]]]:
    """
    Validate DataFrame column headers against the schema and group them by facet.

    Enforces the following validation rules:
    1. Every facet column must follow the format ``"facet:type:field"`` with exactly
       three colon-separated components, or ``"group.N.ids"`` for groupings.
    2. All columns belonging to the same facet must share a single type discriminator tag.
    3. The facet name and type tag must exist in the schema registry.
    4. The field name must exist on the type's msgspec Struct definition.

    Parameters
    ----------
    columns : list of str
        DataFrame column names to validate.
    registry_or_version : dict of str to Any or str
        Schema registry dictionary or schema version string (e.g. ``"1.0"``).

    Returns
    -------
    tuple of (dict[str, list[tuple[str, str, str]]], list[tuple[int, str]])
        Mapping from facet name (`"extent"`, `"payload"`, `"confidence"`) to
        lists of ``(type_tag, field_name, full_col_name)`` tuples, and a list of
        ``(group_index, full_col_name)`` tuples for grouping columns.

    Raises
    ------
    BoppValidationError
        If any of the structural or schema validation rules are violated.
    """
    if isinstance(registry_or_version, str):
        registry = get_registry(registry_or_version)
    else:
        registry = registry_or_version

    facet_types: dict[str, str] = {}
    grouped_cols: dict[str, list[tuple[str, str, str]]] = {
        k: [] for k in FACET_REGISTRY_KEYS
    }
    grouping_cols: list[tuple[int, str]] = []

    for col in columns:
        # Check for grouping column: group.N.ids
        if col.startswith("group."):
            dot_parts = col.split(".")
            if (
                len(dot_parts) == 3
                and dot_parts[0] == "group"
                and dot_parts[1].isdigit()
                and dot_parts[1] == str(int(dot_parts[1]))
                and dot_parts[2] == "ids"
            ):
                grouping_cols.append((int(dot_parts[1]), col))
                continue
            raise BoppValidationError(
                f"Invalid grouping column format '{col}'. Expected 'group.N.ids' with a canonical non-negative integer index N."
            )

        parts = col.split(":")
        # Rule 1: Must have exactly 3 fields (facet:type:field)
        if len(parts) != 3:
            raise BoppValidationError(
                f"Invalid column format '{col}'. Expected exactly 3 colon-separated fields 'facet:type:field' or 'group.N.ids'."
            )

        facet, type_tag, field_name = parts[0], parts[1], parts[2]

        # Rule 2: Single type per facet
        if facet in facet_types and facet_types[facet] != type_tag:
            raise BoppValidationError(
                f"Mixed types for facet '{facet}': found '{facet_types[facet]}' and '{type_tag}'."
            )
        facet_types[facet] = type_tag

        # Rule 3: Valid facet and type in registry
        if facet not in FACET_REGISTRY_KEYS:
            raise BoppValidationError(
                f"Invalid facet '{facet}' in column '{col}'. Valid facets are {list(FACET_REGISTRY_KEYS.keys())}."
            )

        reg_key = FACET_REGISTRY_KEYS[facet]
        type_registry = registry[reg_key]
        if type_tag not in type_registry:
            raise BoppValidationError(
                f"Unrecognized type '{type_tag}' for facet '{facet}' in column '{col}'."
            )

        # Rule 4: Field exists on target type struct
        struct_cls = type_registry[type_tag]
        if field_name not in struct_cls.__struct_fields__:
            raise BoppValidationError(
                f"Field '{field_name}' in column '{col}' does not exist on type '{type_tag}'."
            )

        grouped_cols[facet].append((type_tag, field_name, col))

    grouping_cols.sort(key=lambda item: item[0])
    expected_indices = list(range(len(grouping_cols)))
    actual_indices = [idx for idx, _ in grouping_cols]
    if actual_indices != expected_indices:
        raise BoppValidationError(
            f"Grouping column indices must be contiguous starting from 0 (expected {expected_indices}, got {actual_indices})."
        )
    return grouped_cols, grouping_cols


_NULL_TYPES = (type(None), msgspec.UnsetType)


def _strip_type_wrappers(hint: Any) -> Any:
    """
    Strip ``Annotated`` wrappers and PEP 695 type aliases from a type hint.

    Generated models declare constrained items as aliases, e.g.
    ``type DurationItem = Annotated[float, Meta(ge=0.0)]``.
    """
    while True:
        if get_origin(hint) is typing.Annotated:
            hint = get_args(hint)[0]
        elif isinstance(hint, typing.TypeAliasType):
            hint = hint.__value__
        else:
            return hint


def _union_arms(hint: Any) -> list[Any]:
    """
    Flatten a type hint into its non-null union arms.

    Wrappers and aliases are stripped at every level, nested unions (including unions of
    aliases) are flattened, and ``None``/``UnsetType`` arms are dropped. A hint that is not
    a union yields a single arm.
    """
    hint = _strip_type_wrappers(hint)
    if get_origin(hint) not in (typing.Union, types.UnionType):
        return [hint]
    arms: list[Any] = []
    for arg in get_args(hint):
        if arg not in _NULL_TYPES:
            arms.extend(_union_arms(arg))
    return arms


def _column_item_type(field_type: Any) -> Any | None:
    """
    Return the item type ``T`` of a columnar field ``list[T]`` or ``ArrowArray[DType, T]``.

    Optional columns (``list[T] | UnsetType``, ``ArrowArray[...] | None``) are accepted.
    Returns None if the field is not a list/array column.
    """
    arms = _union_arms(field_type)
    if len(arms) != 1:
        return None
    origin = get_origin(arms[0])
    if origin is list:
        args = get_args(arms[0])
        return args[0] if args else Any
    if getattr(origin, "__name__", "") == "ArrowArray":
        args = get_args(arms[0])
        return args[1] if len(args) >= 2 else Any
    return None


def _scalar_kind(tp: Any) -> str | None:
    """Classify a scalar type as "bool", "int", "float" or "str" (subclasses such as enums included)."""
    if not isinstance(tp, type):
        return None
    # bool before int: bool is a subclass of int
    for base, kind in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str")):
        if issubclass(tp, base):
            return kind
    return None


def _polars_item_dtype(item_type: Any, pl: Any) -> Any | None:
    """
    Map a column item type to a single Polars DataType.

    Nullable items map to the dtype of their non-null arms. Mixed ``int``/``float`` unions
    map to ``Float64``. Returns None if the items have no single scalar dtype (containers,
    ``Any``, or unions mixing strings and numbers), leaving the column to Polars inference.
    """
    kinds: set[str | None] = set()
    for arm in _union_arms(item_type):
        if get_origin(arm) is Literal:
            kinds.update(_scalar_kind(type(value)) for value in get_args(arm))
        else:
            kinds.add(_scalar_kind(arm))

    if kinds == {"int"}:
        return pl.Int64
    if kinds in ({"float"}, {"int", "float"}):
        return pl.Float64
    if kinds == {"bool"}:
        return pl.Boolean
    if kinds == {"str"}:
        return pl.String
    return None


def _polars_column_dtype(field_type: Any, pl: Any) -> Any | None:
    """
    Map a columnar struct field type (e.g. ``list[float]`` or ``ArrowArray[Literal['float32'], float]``) to a Polars DataType.

    Shared by DataFrame construction and CSV reading, so that a column is written and
    read back with the same dtype. Returns None for non-list fields and for items
    without a single scalar dtype.
    """
    arms = _union_arms(field_type)
    if len(arms) == 1:
        origin = get_origin(arms[0])
        if getattr(origin, "__name__", "") == "ArrowArray":
            args = get_args(arms[0])
            if args:
                dtype_arg = args[0]
                dtype_name = (
                    get_args(dtype_arg)[0]
                    if get_origin(dtype_arg) is not None
                    else str(dtype_arg)
                )
                if isinstance(dtype_name, str):
                    dtype_name = dtype_name.strip("'\"")
                mapping = {
                    "float32": pl.Float32,
                    "float64": pl.Float64,
                    "uint8": pl.UInt8,
                    "uint16": pl.UInt16,
                    "uint32": pl.UInt32,
                    "int32": pl.Int32,
                    "bool": pl.Boolean,
                    "string": pl.String,
                }
                if dtype_name in mapping:
                    return mapping[dtype_name]

    item_type = _column_item_type(field_type)
    if item_type is None:
        return None
    return _polars_item_dtype(item_type, pl)


def _extract_facet_data(
    struct: msgspec.Struct | None,
    facet_name: str,
    target_dict: dict[str, Any],
    attrs_dict: dict[str, Any],
    *,
    encode_complex: bool = False,
    complex_fields: set[str] | None = None,
    column_types: dict[str, tuple[Any, bool]] | None = None,
) -> None:
    """
    Extract parallel array fields and scalar attributes from a facet struct.

    Parallel list fields are formatted into column keys (``"facet:tag:field"``)
    and stored in `target_dict` for DataFrame construction. Scalar attributes
    are converted to built-in Python types and stored in `attrs_dict` for DataFrame
    attribute and TOML frontmatter serialization.

    Parameters
    ----------
    struct : msgspec.Struct or None
        The facet struct instance (e.g. payload, extent, confidence) to inspect.
    facet_name : str
        Facet identifier (e.g., ``"payload"``, ``"extent"``, or ``"confidence"``).
    target_dict : dict of str to Any
        Dictionary accumulating columnar lists under ``"facet:tag:field"`` keys.
    attrs_dict : dict of str to Any
        Dictionary accumulating scalar singleton attributes for the facet.
    encode_complex : bool, default False
        Whether to JSON-encode nested/complex column elements to strings.
    complex_fields : set of str or None, default None
        Explicit names of fields that contain complex/nested structures.
    column_types : dict of str to tuple of (Any, bool) or None, default None
        Dictionary mapping column names to (field_type, is_encoded_complex).
    """
    if struct is None or struct is msgspec.UNSET:
        return

    tag = _get_tag(struct)
    type_field_name = f"{facet_name}_type"

    facet_attrs: dict[str, Any] = {}
    has_list_fields = False

    for field in _get_struct_fields(type(struct)):
        if field.name == type_field_name:
            continue
        val = getattr(struct, field.name)
        if val is None or val is msgspec.UNSET:
            continue

        if field.name.startswith(("min_", "max_")) or field.name in (
            "disjoint",
            "exhaustive",
        ):
            facet_attrs[field.name] = msgspec.to_builtins(val)
            continue

        if isinstance(val, (list, pa.Array, pa.ChunkedArray)):
            has_list_fields = True
            col_key = f"{facet_name}:{tag}:{field.name}"
            is_complex = False
            if encode_complex:
                is_complex = (
                    complex_fields is not None and field.name in complex_fields
                ) or any(
                    isinstance(item, (list, dict, tuple, msgspec.Struct))
                    for item in (val.to_pylist() if hasattr(val, "to_pylist") else val)
                    if item is not None
                )
                if is_complex:
                    items = val.to_pylist() if hasattr(val, "to_pylist") else val
                    target_dict[col_key] = [
                        msgspec.json.encode(item).decode("utf-8")
                        if item is not None and item is not msgspec.UNSET
                        else None
                        for item in items
                    ]
                else:
                    target_dict[col_key] = val
            else:
                target_dict[col_key] = val

            if column_types is not None:
                column_types[col_key] = (field.type, is_complex)
        else:
            facet_attrs[field.name] = msgspec.to_builtins(val)

    if not has_list_fields and tag is not None:
        facet_attrs[type_field_name] = tag

    if facet_attrs:
        attrs_dict[facet_name] = facet_attrs


def to_dataframe(
    annotation: BoppBase,
    backend: Literal["pandas", "pandas-pyarrow", "polars"] = "pandas",
    *,
    encode_complex: bool = False,
) -> AnnotationDataFrame:
    """
    Convert an Annotation instance into a DataFrame and separate header dictionary.

    Returns an :class:`AnnotationDataFrame` namedtuple containing the DataFrame
    (``df``) with parallel columnar arrays and the header dictionary (``header``)
    containing singleton metadata and scalar facet attributes.

    Parameters
    ----------
    annotation : BoppBase
        The Annotation struct instance to convert.
    backend : {"pandas", "pandas-pyarrow", "polars"}, default "pandas"
        The DataFrame framework and backend to return.
    encode_complex : bool, default False
        Whether to JSON-encode nested/complex columns (e.g. lists of fractions or dicts)
        into strings. Set to True when serializing to tabular formats like CSV.

    Returns
    -------
    AnnotationDataFrame
        Namedtuple with ``df`` (Pandas or Polars DataFrame) and ``header`` (dict of metadata).

    See Also
    --------
    from_dataframe : Reconstitute an Annotation struct from a DataFrame and header.

    Examples
    --------
    Direct unpacking into dataframe and header objects:

    >>> from bopp.models.v1.annotation import Annotation
    >>> ann = Annotation(
    ...     media_id="audio:123",
    ...     payload={"payload_type": "beat", "value": [1.0, 2.0]}
    ... )
    >>> df, header = to_dataframe(ann, backend="pandas")
    >>> header["media_id"]
    'audio:123'
    >>> df.columns.tolist()
    ['payload:beat:value']

    Accessing attributes via named fields:

    >>> ann_df = to_dataframe(ann, backend="pandas")
    >>> ann_df.header["media_id"]
    'audio:123'
    >>> ann_df.df.shape
    (2, 1)
    """
    data: dict[str, Any] = {}
    bopp_version = getattr(annotation, "bopp_version", get_current_schema_version())

    header: dict[str, Any] = {
        "bopp_version": bopp_version,
        "media_id": getattr(annotation, "media_id", None),
    }

    complex_registry: dict[str, dict[str, list[str]]] = {}
    if encode_complex:
        try:
            registry = get_registry(bopp_version)
            complex_registry = registry.get("COMPLEX_FIELDS_REGISTRY", {})
        except (BoppRegistryError, ValueError):
            complex_registry = {}

    def _get_complex_fields_for_facet(
        struct: msgspec.Struct | None, facet_name: str
    ) -> set[str]:
        if struct is None or struct is msgspec.UNSET:
            return set()
        tag = _get_tag(struct)
        if not tag:
            return set()
        tag_key = f"{facet_name}_type"
        return set(complex_registry.get(tag_key, {}).get(tag, []))

    column_types: dict[str, tuple[Any, bool]] = {}

    for facet_name in ("payload", "extent", "confidence"):
        facet_struct = getattr(annotation, facet_name, None)
        _extract_facet_data(
            facet_struct,
            facet_name,
            data,
            header,
            encode_complex=encode_complex,
            complex_fields=_get_complex_fields_for_facet(facet_struct, facet_name),
            column_types=column_types,
        )

    # Process groupings
    groupings_val = getattr(annotation, "groupings", msgspec.UNSET)
    if groupings_val not in (msgspec.UNSET, None) and isinstance(groupings_val, list):
        groupings_header = []
        for idx, g in enumerate(groupings_val):
            col_key = f"group.{idx}.ids"
            data[col_key] = getattr(g, "ids", [])
            column_types[col_key] = (list[int], False)

            g_dict: dict[str, Any] = {
                "name": g.name,
                "disjoint": getattr(g, "disjoint", False),
                "exhaustive": getattr(g, "exhaustive", False),
            }
            labels = getattr(g, "labels", msgspec.UNSET)
            if labels not in (msgspec.UNSET, None):
                g_dict["labels"] = msgspec.to_builtins(labels)
            groupings_header.append(g_dict)

        if groupings_header:
            header["groupings"] = groupings_header

    for field_name in annotation.__struct_fields__:
        if field_name in FACET_REGISTRY_KEYS or field_name in (
            "bopp_version",
            "media_id",
            "groupings",
        ):
            continue
        val = getattr(annotation, field_name, msgspec.UNSET)
        if val not in (msgspec.UNSET, None):
            header[field_name] = msgspec.to_builtins(val)

    if backend in ("pandas", "pandas-pyarrow"):
        try:
            import pandas as pd
        except ImportError as err:
            raise BoppIOError("pandas is required for backend='pandas'") from err

        if backend == "pandas-pyarrow":
            try:
                import pyarrow  # noqa: F401
            except ImportError as err:
                raise BoppIOError(
                    "pyarrow is required for backend='pandas-pyarrow'"
                ) from err

            pd_data = {
                k: pd.Series(pd.arrays.ArrowExtensionArray(v))
                if isinstance(v, (pa.Array, pa.ChunkedArray))
                else v
                for k, v in data.items()
            }
            df_pd = pd.DataFrame(pd_data).convert_dtypes(dtype_backend="pyarrow")
        else:
            pd_data = {
                k: v.to_numpy(zero_copy_only=False)
                if isinstance(v, (pa.Array, pa.ChunkedArray))
                and not pa.types.is_string(v.type)
                else (v.to_pylist() if hasattr(v, "to_pylist") else v)
                for k, v in data.items()
            }
            df_pd = pd.DataFrame(pd_data)

        return AnnotationDataFrame(df=df_pd, header=header)

    elif backend == "polars":
        schema: dict[str, Any] = {}
        for col_name, (field_type, is_complex) in column_types.items():
            # JSON-encoded complex cells are strings
            pl_dtype = pl.String if is_complex else _polars_column_dtype(field_type, pl)
            if pl_dtype is not None:
                schema[col_name] = pl_dtype

        if len(schema) == len(data):
            df_pl = pl.DataFrame(data, schema=schema)
        else:
            df_pl = pl.DataFrame(data, schema_overrides=schema if schema else None)
        return AnnotationDataFrame(df=df_pl, header=header)

    else:
        raise BoppArgumentError(f"Unsupported backend: {backend}")


def from_dataframe(
    df_or_tuple: Any,
    header: dict[str, Any] | None = None,
) -> BoppBase:
    """
    Reconstitute a strictly typed Annotation struct from a DataFrame and header metadata.

    Accepts the input either as a combined :class:`AnnotationDataFrame` / 2-tuple
    ``(df, header)``, or as separate ``df`` and ``header`` arguments.

    Parameters
    ----------
    df_or_tuple : pandas.DataFrame, polars.DataFrame, AnnotationDataFrame, or tuple
        DataFrame containing columnar arrays, or an ``AnnotationDataFrame`` / 2-tuple of
        ``(df, header)``.
    header : dict of str to Any or None, optional
        Metadata header dictionary containing singleton fields (e.g. ``media_id``,
        ``bopp_version``, scalar facet attributes). If None, unpacked from `df_or_tuple`.

    Returns
    -------
    BoppBase
        Validated Annotation struct built from the DataFrame data and header.

    See Also
    --------
    to_dataframe : Convert an Annotation instance into an AnnotationDataFrame.

    Examples
    --------
    Passing an AnnotationDataFrame namedtuple or 2-tuple directly:

    >>> ann_df = to_dataframe(ann)
    >>> ann = from_dataframe(ann_df)

    Unpacking and passing as separate arguments:

    >>> df, header = to_dataframe(ann)
    >>> ann = from_dataframe(df, header)
    """
    if header is None:
        if isinstance(df_or_tuple, tuple) and len(df_or_tuple) == 2:
            df, header = df_or_tuple[0], df_or_tuple[1]
        else:
            df = df_or_tuple
            header = {}
    else:
        df = df_or_tuple

    # Confirm pandas/polars type dynamically
    valid = isinstance(df, pl.DataFrame)

    if not valid:
        try:
            import pandas as pd

            valid = isinstance(df, pd.DataFrame)
        except ImportError:
            valid = False

    if not valid:
        raise BoppArgumentError(
            f"Unsupported DataFrame type: {type(df)}. Must be a pandas or polars DataFrame."
        )

    columns = list(df.columns)
    bopp_version = header.get("bopp_version", get_current_schema_version())

    registry = get_registry(bopp_version)
    annotation_cls = registry["Annotation"]

    grouped_cols, grouping_cols = _validate_dataframe_columns(columns, registry)

    media_id = header.get("media_id")
    if not media_id or not isinstance(media_id, str):
        raise BoppValidationError(
            "Missing or invalid required 'media_id' in DataFrame header attributes."
        )

    bopp_data: dict[str, Any] = {
        "bopp_version": bopp_version,
        "media_id": media_id,
    }

    for key in ("metadata", "sandbox", "id", "parents"):
        if key in header:
            bopp_data[key] = header[key]

    complex_registry: dict[str, dict[str, list[str]]] = registry.get(
        "COMPLEX_FIELDS_REGISTRY", {}
    )

    def _get_column_data(col_name: str, is_complex: bool, as_arrow: bool = False) -> Any:
        col_series = df[col_name]
        if is_complex or not as_arrow:
            extractor = getattr(
                col_series, "to_list", getattr(col_series, "tolist", None)
            )
            return extractor() if extractor is not None else list(col_series)
        if hasattr(col_series, "to_arrow"):
            return col_series.to_arrow()
        try:
            return pa.array(col_series)
        except Exception:
            extractor = getattr(
                col_series, "to_list", getattr(col_series, "tolist", None)
            )
            return extractor() if extractor is not None else list(col_series)

    for facet, cols in grouped_cols.items():
        facet_header = header.get(facet)
        if not cols:
            if isinstance(facet_header, dict) and f"{facet}_type" in facet_header:
                bopp_data[facet] = dict(facet_header)
            continue

        facet_type = cols[0][0]
        facet_dict: dict[str, Any] = {f"{facet}_type": facet_type}

        if isinstance(facet_header, dict):
            facet_dict.update(facet_header)

        facet_registry_key = FACET_REGISTRY_KEYS.get(facet)
        struct_cls = (
            registry.get(facet_registry_key, {}).get(facet_type)
            if facet_registry_key
            else None
        )
        try:
            hints = (
                get_type_hints(struct_cls, include_extras=True)
                if struct_cls
                else {}
            )
        except Exception:
            hints = getattr(struct_cls, "__annotations__", {}) if struct_cls else {}

        for _, field_name, full_col_name in cols:
            is_complex = field_name in complex_registry.get(
                f"{facet}_type", {}
            ).get(facet_type, [])
            hint = hints.get(field_name)
            is_arrow = unwrap_arrow_array(hint) is not None if hint is not None else False
            facet_dict[field_name] = _get_column_data(
                full_col_name, is_complex, as_arrow=is_arrow
            )

        if facet == "extent":
            facet_dict.setdefault("disjoint", facet_type == "time_uniform")
            facet_dict.setdefault("exhaustive", False)

        bopp_data[facet] = facet_dict

    if grouping_cols:
        groupings_header_list = header.get("groupings", [])
        grouping_items = []
        for g_idx, col_name in grouping_cols:
            g_meta = (
                groupings_header_list[g_idx]
                if g_idx < len(groupings_header_list)
                else {}
            )
            g_dict = {
                "name": g_meta.get("name", f"group_{g_idx}"),
                "ids": _get_column_data(col_name, is_complex=False, as_arrow=True),
                "disjoint": g_meta.get("disjoint", False),
                "exhaustive": g_meta.get("exhaustive", False),
            }
            if "labels" in g_meta:
                g_dict["labels"] = g_meta["labels"]
            grouping_items.append(g_dict)
        bopp_data["groupings"] = grouping_items

    return msgspec.convert(bopp_data, type=annotation_cls, dec_hook=arrow_dec_hook)


# ==========================================
# 2. Extent Topology & Bounds Utilities
# ==========================================

EXTENT_AXES_METADATA: dict[str, list[dict[str, Any]]] = {
    "time": [
        {
            "dim": "time",
            "kind": "point",
            "field": "time",
            "min_attr": "min_time",
            "max_attr": "max_time",
        }
    ],
    "time_interval": [
        {
            "dim": "time",
            "kind": "origin_span",
            "origin": "time",
            "span": "duration",
            "min_attr": "min_time",
            "max_attr": "max_time",
        }
    ],
    "time_uniform": [
        {
            "dim": "time",
            "kind": "uniform",
            "min_attr": "min_time",
            "max_attr": "max_time",
        }
    ],
    "time_frequency_box": [
        {
            "dim": "time",
            "kind": "origin_span",
            "origin": "time",
            "span": "duration",
            "min_attr": "min_time",
            "max_attr": "max_time",
        },
        {
            "dim": "frequency",
            "kind": "min_max",
            "min_field": "freq_min",
            "max_field": "freq_max",
            "min_attr": "min_freq",
            "max_attr": "max_freq",
        },
    ],
    "pixel": [
        {
            "dim": "x",
            "kind": "point",
            "field": "x",
            "min_attr": "min_x",
            "max_attr": "max_x",
        },
        {
            "dim": "y",
            "kind": "point",
            "field": "y",
            "min_attr": "min_y",
            "max_attr": "max_y",
        },
    ],
    "pixel_box": [
        {
            "dim": "x",
            "kind": "origin_span",
            "origin": "x",
            "span": "width",
            "min_attr": "min_x",
            "max_attr": "max_x",
        },
        {
            "dim": "y",
            "kind": "origin_span",
            "origin": "y",
            "span": "height",
            "min_attr": "min_y",
            "max_attr": "max_y",
        },
    ],
    "midi_tick": [
        {
            "dim": "tick",
            "kind": "point",
            "field": "tick",
            "min_attr": "min_tick",
            "max_attr": "max_tick",
        }
    ],
    "midi_interval": [
        {
            "dim": "tick",
            "kind": "origin_span",
            "origin": "tick",
            "span": "duration",
            "min_attr": "min_tick",
            "max_attr": "max_tick",
        }
    ],
    "score_quarter": [
        {
            "dim": "quarter",
            "kind": "point",
            "field": "quarter",
            "min_attr": "min_quarter",
            "max_attr": "max_quarter",
        }
    ],
    "score_quarter_float": [
        {
            "dim": "quarter",
            "kind": "point",
            "field": "quarter",
            "min_attr": "min_quarter",
            "max_attr": "max_quarter",
        }
    ],
    "score_interval": [
        {
            "dim": "quarter",
            "kind": "origin_span",
            "origin": "quarter",
            "span": "duration",
            "min_attr": "min_quarter",
            "max_attr": "max_quarter",
        }
    ],
    "score_interval_float": [
        {
            "dim": "quarter",
            "kind": "origin_span",
            "origin": "quarter",
            "span": "duration",
            "min_attr": "min_quarter",
            "max_attr": "max_quarter",
        }
    ],
    "text": [
        {
            "dim": "offset",
            "kind": "point",
            "field": "offset",
            "min_attr": "min_offset",
            "max_attr": "max_offset",
        }
    ],
    "text_interval": [
        {
            "dim": "offset",
            "kind": "origin_span",
            "origin": "offset",
            "span": "length",
            "min_attr": "min_offset",
            "max_attr": "max_offset",
        }
    ],
    "cartesian": [
        {
            "dim": "x",
            "kind": "point",
            "field": "x",
            "min_attr": "min_x",
            "max_attr": "max_x",
        },
        {
            "dim": "y",
            "kind": "point",
            "field": "y",
            "min_attr": "min_y",
            "max_attr": "max_y",
        },
        {
            "dim": "z",
            "kind": "point",
            "field": "z",
            "min_attr": "min_z",
            "max_attr": "max_z",
        },
    ],
    "cartesian_time": [
        {
            "dim": "time",
            "kind": "point",
            "field": "time",
            "min_attr": "min_time",
            "max_attr": "max_time",
        },
        {
            "dim": "x",
            "kind": "point",
            "field": "x",
            "min_attr": "min_x",
            "max_attr": "max_x",
        },
        {
            "dim": "y",
            "kind": "point",
            "field": "y",
            "min_attr": "min_y",
            "max_attr": "max_y",
        },
        {
            "dim": "z",
            "kind": "point",
            "field": "z",
            "min_attr": "min_z",
            "max_attr": "max_z",
        },
    ],
    "cartesian_time_interval": [
        {
            "dim": "time",
            "kind": "origin_span",
            "origin": "time",
            "span": "duration",
            "min_attr": "min_time",
            "max_attr": "max_time",
        },
        {
            "dim": "x",
            "kind": "point",
            "field": "x",
            "min_attr": "min_x",
            "max_attr": "max_x",
        },
        {
            "dim": "y",
            "kind": "point",
            "field": "y",
            "min_attr": "min_y",
            "max_attr": "max_y",
        },
        {
            "dim": "z",
            "kind": "point",
            "field": "z",
            "min_attr": "min_z",
            "max_attr": "max_z",
        },
    ],
    "spherical": [
        {
            "dim": "azimuth",
            "kind": "point",
            "field": "azimuth",
            "min_attr": "min_azimuth",
            "max_attr": "max_azimuth",
        },
        {
            "dim": "elevation",
            "kind": "point",
            "field": "elevation",
            "min_attr": "min_elevation",
            "max_attr": "max_elevation",
        },
        {
            "dim": "radius",
            "kind": "point",
            "field": "radius",
            "min_attr": "min_radius",
            "max_attr": "max_radius",
        },
    ],
    "spherical_time": [
        {
            "dim": "time",
            "kind": "point",
            "field": "time",
            "min_attr": "min_time",
            "max_attr": "max_time",
        },
        {
            "dim": "azimuth",
            "kind": "point",
            "field": "azimuth",
            "min_attr": "min_azimuth",
            "max_attr": "max_azimuth",
        },
        {
            "dim": "elevation",
            "kind": "point",
            "field": "elevation",
            "min_attr": "min_elevation",
            "max_attr": "max_elevation",
        },
        {
            "dim": "radius",
            "kind": "point",
            "field": "radius",
            "min_attr": "min_radius",
            "max_attr": "max_radius",
        },
    ],
    "spherical_time_interval": [
        {
            "dim": "time",
            "kind": "origin_span",
            "origin": "time",
            "span": "duration",
            "min_attr": "min_time",
            "max_attr": "max_time",
        },
        {
            "dim": "azimuth",
            "kind": "point",
            "field": "azimuth",
            "min_attr": "min_azimuth",
            "max_attr": "max_azimuth",
        },
        {
            "dim": "elevation",
            "kind": "point",
            "field": "elevation",
            "min_attr": "min_elevation",
            "max_attr": "max_elevation",
        },
        {
            "dim": "radius",
            "kind": "point",
            "field": "radius",
            "min_attr": "min_radius",
            "max_attr": "max_radius",
        },
    ],
}


FRACTION_EXTENT_TAGS: frozenset[str] = frozenset({"score_quarter", "score_interval"})


def _to_coord(val: Any) -> Any:
    """Normalize fractional pairs to Fraction and pass numbers through unchanged."""
    if (
        isinstance(val, (list, tuple))
        and len(val) == 2
        and isinstance(val[0], int)
        and isinstance(val[1], int)
    ):
        return Fraction(val[0], val[1])
    return val


def _normalize_coords(vals: Any, is_fraction: bool) -> list[Any]:
    """Normalize coordinate array elements only if the extent uses rational fractions."""
    if hasattr(vals, "to_pylist"):
        vals = vals.to_pylist()
    if not is_fraction:
        return vals if isinstance(vals, list) else list(vals)
    return [Fraction(v[0], v[1]) if isinstance(v, (list, tuple)) else v for v in vals]


def _partition_group_indices(ids: Any) -> dict[int, list[int]]:
    """Partition observation indices by non-negative group ID in a single O(N) pass."""
    if hasattr(ids, "to_pylist"):
        ids = ids.to_pylist()
    groups: dict[int, list[int]] = defaultdict(list)
    for i, gid in enumerate(ids):
        if gid >= 0:
            groups[gid].append(i)
    return groups


def _resolve_extent_and_n(
    extent_or_annotation: Any,
    N: int | None = None,
) -> tuple[msgspec.Struct | None, int | None]:
    """Extract extent struct and observation count from an Extent or Annotation."""
    if extent_or_annotation is None or extent_or_annotation is msgspec.UNSET:
        return None, 0
    if isinstance(extent_or_annotation, BoppBase):
        extent = getattr(extent_or_annotation, "extent", None)
        if N is None:
            lengths = extent_or_annotation._get_all_column_lengths(
                getattr(extent_or_annotation, "payload", None), "Payload"
            )
            N = lengths[0] if lengths else 0
        return extent, N
    return extent_or_annotation, N


def infer_extent_bounds(
    extent_or_annotation: Any,
    *,
    indices: list[int] | None = None,
    N: int | None = None,
) -> dict[str, tuple[Any, Any]]:
    """
    Infer the minimum and maximum boundaries along each dimension from extent observations.

    Parameters
    ----------
    extent_or_annotation : msgspec.Struct or BoppBase
        An extent struct or parent annotation instance.
    indices : list of int or None, optional
        Subset of observation indices to calculate bounds over (e.g. for grouping evaluation).
    N : int or None, optional
        Total number of observations (used for time_uniform extents).

    Returns
    -------
    dict of str to tuple of (Any, Any)
        Mapping from dimension name to (inferred_min, inferred_max).
    """
    extent, N = _resolve_extent_and_n(extent_or_annotation, N)
    if extent is None or extent is msgspec.UNSET:
        return {}

    tag = _get_tag(extent)
    specs = EXTENT_AXES_METADATA.get(tag or "", [])
    is_fraction = tag in FRACTION_EXTENT_TAGS
    bounds: dict[str, tuple[Any, Any]] = {}

    for spec in specs:
        dim = spec["dim"]
        kind = spec["kind"]

        if kind == "point":
            raw_vals = getattr(extent, spec["field"], [])
            if hasattr(raw_vals, "to_pylist"):
                raw_vals = raw_vals.to_pylist()
            if indices is not None:
                raw_vals = [raw_vals[i] for i in indices]
            if not raw_vals:
                bounds[dim] = (None, None)
            else:
                coords = _normalize_coords(raw_vals, is_fraction)
                bounds[dim] = (min(coords), max(coords))

        elif kind == "origin_span":
            raw_starts = getattr(extent, spec["origin"], [])
            raw_spans = getattr(extent, spec["span"], [])
            if hasattr(raw_starts, "to_pylist"):
                raw_starts = raw_starts.to_pylist()
            if hasattr(raw_spans, "to_pylist"):
                raw_spans = raw_spans.to_pylist()
            if indices is not None:
                raw_starts = [raw_starts[i] for i in indices]
                raw_spans = [raw_spans[i] for i in indices]
            if not raw_starts:
                bounds[dim] = (None, None)
            else:
                starts = _normalize_coords(raw_starts, is_fraction)
                spans = _normalize_coords(raw_spans, is_fraction)
                bounds[dim] = (
                    min(starts),
                    max(s + sp for s, sp in zip(starts, spans)),
                )

        elif kind == "min_max":
            raw_mins = getattr(extent, spec["min_field"], [])
            raw_maxs = getattr(extent, spec["max_field"], [])
            if hasattr(raw_mins, "to_pylist"):
                raw_mins = raw_mins.to_pylist()
            if hasattr(raw_maxs, "to_pylist"):
                raw_maxs = raw_maxs.to_pylist()
            if indices is not None:
                raw_mins = [raw_mins[i] for i in indices]
                raw_maxs = [raw_maxs[i] for i in indices]
            if not raw_mins:
                bounds[dim] = (None, None)
            else:
                mins = _normalize_coords(raw_mins, is_fraction)
                maxs = _normalize_coords(raw_maxs, is_fraction)
                bounds[dim] = (min(mins), max(maxs))

        elif kind == "uniform":
            start_time = getattr(extent, "start_time", 0.0)
            sampling_period = getattr(extent, "sampling_period", 1.0)
            count = len(indices) if indices is not None else (N if N is not None else 0)
            if count <= 0:
                bounds[dim] = (None, None)
            elif count == 1:
                bounds[dim] = (start_time, start_time)
            else:
                bounds[dim] = (
                    start_time,
                    start_time + (count - 1) * sampling_period,
                )

    return bounds


def _get_explicit_extent_bounds(
    extent: msgspec.Struct,
) -> dict[str, tuple[Any, Any]]:
    """Extract only explicitly specified min/max attributes from an extent."""
    tag = _get_tag(extent)
    specs = EXTENT_AXES_METADATA.get(tag or "", [])
    result: dict[str, tuple[Any, Any]] = {}
    for spec in specs:
        min_attr = getattr(extent, spec["min_attr"], msgspec.UNSET)
        max_attr = getattr(extent, spec["max_attr"], msgspec.UNSET)
        b_min = (
            None
            if (min_attr is msgspec.UNSET or min_attr is None)
            else _to_coord(min_attr)
        )
        b_max = (
            None
            if (max_attr is msgspec.UNSET or max_attr is None)
            else _to_coord(max_attr)
        )
        result[spec["dim"]] = (b_min, b_max)
    return result


def get_extent_bounds(
    extent_or_annotation: Any,
    *,
    indices: list[int] | None = None,
    N: int | None = None,
) -> dict[str, tuple[Any, Any]]:
    """
    Retrieve effective boundaries for an extent, combining explicit min/max bounds with inferred bounds.

    Parameters
    ----------
    extent_or_annotation : msgspec.Struct or BoppBase
        An extent struct or parent annotation instance.
    indices : list of int or None, optional
        Subset of observation indices.
    N : int or None, optional
        Total number of observations.

    Returns
    -------
    dict of str to tuple of (Any, Any)
        Mapping from dimension name to effective (bound_min, bound_max).
    """
    extent, N = _resolve_extent_and_n(extent_or_annotation, N)
    if extent is None or extent is msgspec.UNSET:
        return {}

    explicit = _get_explicit_extent_bounds(extent)
    if all(
        b_min is not None and b_max is not None for b_min, b_max in explicit.values()
    ):
        return explicit

    inferred = infer_extent_bounds(extent, indices=indices, N=N)
    result: dict[str, tuple[Any, Any]] = {}

    for dim, (exp_min, exp_max) in explicit.items():
        inf_min, inf_max = inferred.get(dim, (None, None))
        result[dim] = (
            inf_min if exp_min is None else exp_min,
            inf_max if exp_max is None else exp_max,
        )

    return result


def _union_area_2d(
    rects: list[tuple[float, float, float, float]],
    bounds: tuple[float, float, float, float],
) -> float:
    """Compute the total covered area of 2D axis-aligned rectangles clipped to a bounding box."""
    xmin, xmax, ymin, ymax = bounds
    clipped = []
    for x1, x2, y1, y2 in rects:
        cx1 = max(xmin, min(xmax, x1))
        cx2 = max(xmin, min(xmax, x2))
        cy1 = max(ymin, min(ymax, y1))
        cy2 = max(ymin, min(ymax, y2))
        if cx2 > cx1 and cy2 > cy1:
            clipped.append((cx1, cx2, cy1, cy2))
    if not clipped:
        return 0.0

    y_coords = sorted(
        set([cy1 for _, _, cy1, _ in clipped] + [cy2 for _, _, _, cy2 in clipped])
    )
    total_area = 0.0

    for i in range(len(y_coords) - 1):
        y_bot = y_coords[i]
        y_top = y_coords[i + 1]
        x_intervals = []
        for x1, x2, y1, y2 in clipped:
            if y1 <= y_bot and y2 >= y_top:
                x_intervals.append((x1, x2))
        if not x_intervals:
            continue
        x_intervals.sort()
        cur_start, cur_end = x_intervals[0]
        x_len = 0.0
        for s, e in x_intervals[1:]:
            if s > cur_end:
                x_len += cur_end - cur_start
                cur_start, cur_end = s, e
            else:
                cur_end = max(cur_end, e)
        x_len += cur_end - cur_start
        total_area += (y_top - y_bot) * x_len

    return total_area


def _is_disjoint_indices(extent: msgspec.Struct, indices: list[int] | None) -> bool:
    """Check disjointedness of an extent over a subset of observation indices."""
    tag = _get_tag(extent)
    if tag == "time_uniform":
        return True

    specs = EXTENT_AXES_METADATA.get(tag or "", [])
    if not specs:
        return True

    is_fraction = tag in FRACTION_EXTENT_TAGS

    # 1D Point extents
    if len(specs) == 1 and specs[0]["kind"] == "point":
        vals = getattr(extent, specs[0]["field"], [])
        if hasattr(vals, "to_pylist"):
            vals = vals.to_pylist()
        if indices is not None:
            vals = [vals[i] for i in indices]
        coords = _normalize_coords(vals, is_fraction)
        return len(set(coords)) == len(coords)

    # 1D Interval extents
    if len(specs) == 1 and specs[0]["kind"] == "origin_span":
        raw_starts = getattr(extent, specs[0]["origin"], [])
        raw_spans = getattr(extent, specs[0]["span"], [])
        if hasattr(raw_starts, "to_pylist"):
            raw_starts = raw_starts.to_pylist()
        if hasattr(raw_spans, "to_pylist"):
            raw_spans = raw_spans.to_pylist()
        if indices is not None:
            raw_starts = [raw_starts[i] for i in indices]
            raw_spans = [raw_spans[i] for i in indices]
        if len(raw_starts) <= 1:
            return True
        starts = _normalize_coords(raw_starts, is_fraction)
        spans = _normalize_coords(raw_spans, is_fraction)
        pairs = sorted(zip(starts, spans))
        prev_s, prev_sp = pairs[0]
        max_end = prev_s + prev_sp
        for s, sp in pairs[1:]:
            if is_fraction or isinstance(s, int):
                if s < max_end:
                    return False
                if s == max_end and sp == 0 and prev_sp == 0:
                    return False
            else:
                if s < max_end and not math.isclose(s, max_end, rel_tol=1e-5, abs_tol=1e-5):
                    return False
                if math.isclose(s, max_end, rel_tol=1e-5, abs_tol=1e-5) and sp == 0 and prev_sp == 0:
                    return False
            max_end = max(max_end, s + sp)
            prev_sp = sp
        return True

    # Multi-dimensional Point extents
    if all(s["kind"] == "point" for s in specs):
        col_lists = []
        for s in specs:
            arr = getattr(extent, s["field"], [])
            if hasattr(arr, "to_pylist"):
                arr = arr.to_pylist()
            if indices is not None:
                arr = [arr[i] for i in indices]
            col_lists.append(arr)
        n = len(col_lists[0])
        return len(set(zip(*col_lists))) == n

    # 2D Box extents (pixel_box, time_frequency_box)
    if tag in ("pixel_box", "time_frequency_box"):
        if tag == "pixel_box":
            xs = getattr(extent, "x", [])
            ws = getattr(extent, "width", [])
            ys = getattr(extent, "y", [])
            hs = getattr(extent, "height", [])
            if hasattr(xs, "to_pylist"):
                xs = xs.to_pylist()
            if hasattr(ws, "to_pylist"):
                ws = ws.to_pylist()
            if hasattr(ys, "to_pylist"):
                ys = ys.to_pylist()
            if hasattr(hs, "to_pylist"):
                hs = hs.to_pylist()
            if indices is not None:
                xs = [xs[i] for i in indices]
                ws = [ws[i] for i in indices]
                ys = [ys[i] for i in indices]
                hs = [hs[i] for i in indices]
            boxes = [(x, x + w, y, y + h) for x, w, y, h in zip(xs, ws, ys, hs)]
        else:
            ts = getattr(extent, "time", [])
            ds = getattr(extent, "duration", [])
            f_mins = getattr(extent, "freq_min", [])
            f_maxs = getattr(extent, "freq_max", [])
            if hasattr(ts, "to_pylist"):
                ts = ts.to_pylist()
            if hasattr(ds, "to_pylist"):
                ds = ds.to_pylist()
            if hasattr(f_mins, "to_pylist"):
                f_mins = f_mins.to_pylist()
            if hasattr(f_maxs, "to_pylist"):
                f_maxs = f_maxs.to_pylist()
            if indices is not None:
                ts = [ts[i] for i in indices]
                ds = [ds[i] for i in indices]
                f_mins = [f_mins[i] for i in indices]
                f_maxs = [f_maxs[i] for i in indices]
            boxes = [
                (t, t + d, f_min, f_max)
                for t, d, f_min, f_max in zip(ts, ds, f_mins, f_maxs)
            ]

        if len(boxes) <= 1:
            return True
        boxes.sort()
        for i in range(len(boxes)):
            b1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                b2 = boxes[j]
                if b2[0] >= b1[1]:
                    break
                if max(b1[0], b2[0]) < min(b1[1], b2[1]) and max(b1[2], b2[2]) < min(
                    b1[3], b2[3]
                ):
                    return False
        return True

    # Spatiotemporal intervals (cartesian_time_interval, spherical_time_interval)
    if tag in ("cartesian_time_interval", "spherical_time_interval"):
        spatial_fields = (
            ("x", "y", "z")
            if tag == "cartesian_time_interval"
            else ("azimuth", "elevation", "radius")
        )
        ts = getattr(extent, "time", [])
        ds = getattr(extent, "duration", [])
        c0, c1, c2 = (getattr(extent, f, []) for f in spatial_fields)
        if hasattr(ts, "to_pylist"):
            ts = ts.to_pylist()
        if hasattr(ds, "to_pylist"):
            ds = ds.to_pylist()
        if hasattr(c0, "to_pylist"):
            c0 = c0.to_pylist()
        if hasattr(c1, "to_pylist"):
            c1 = c1.to_pylist()
        if hasattr(c2, "to_pylist"):
            c2 = c2.to_pylist()
        if indices is not None:
            ts = [ts[i] for i in indices]
            ds = [ds[i] for i in indices]
            c0 = [c0[i] for i in indices]
            c1 = [c1[i] for i in indices]
            c2 = [c2[i] for i in indices]
            ds = [ds[i] for i in indices]
            c0 = [c0[i] for i in indices]
            c1 = [c1[i] for i in indices]
            c2 = [c2[i] for i in indices]

        spatial_groups: dict[tuple[Any, Any, Any], list[tuple[Any, Any]]] = defaultdict(
            list
        )
        for x0, x1, x2, t, d in zip(c0, c1, c2, ts, ds):
            spatial_groups[(x0, x1, x2)].append((t, t + d))

        for intervals in spatial_groups.values():
            if len(intervals) <= 1:
                continue
            intervals.sort()
            max_end = intervals[0][1]
            for s, e in intervals[1:]:
                if s < max_end and not math.isclose(s, max_end, rel_tol=1e-5, abs_tol=1e-5):
                    return False
                max_end = max(max_end, e)
        return True

    return True


def is_disjoint(
    extent_or_annotation: Any,
    *,
    grouping: Any = None,
    group_id: int | None = None,
) -> bool:
    """
    Check whether all observations in an extent or grouping are non-overlapping.

    Parameters
    ----------
    extent_or_annotation : msgspec.Struct or BoppBase
        An extent struct or parent annotation instance.
    grouping : Grouping or None, optional
        Grouping instance to evaluate disjointedness within groups.
    group_id : int or None, optional
        Specific group ID to evaluate. If None and grouping is provided, verifies
        that every non-negative group ID (>= 0) is disjoint.

    Returns
    -------
    bool
        True if all observations are non-overlapping, False otherwise.
    """
    extent, _ = _resolve_extent_and_n(extent_or_annotation)
    if extent is None or extent is msgspec.UNSET:
        return True

    if grouping is not None:
        ids = getattr(grouping, "ids", None)
        if ids is None or ids is msgspec.UNSET:
            return True
        if hasattr(ids, "to_pylist"):
            ids = ids.to_pylist()
        if group_id is not None:
            indices = [i for i, gid in enumerate(ids) if gid == group_id]
            return _is_disjoint_indices(extent, indices)
        else:
            groups = _partition_group_indices(ids)
            for indices in groups.values():
                if not _is_disjoint_indices(extent, indices):
                    return False
            return True

    return _is_disjoint_indices(extent, None)


def _is_exhaustive_indices(
    extent: msgspec.Struct,
    *,
    indices: list[int] | None,
    bounds: dict[str, tuple[Any, Any]],
    N: int | None,
) -> bool:
    """Check exhaustiveness of an extent over a subset of observation indices."""
    tag = _get_tag(extent)
    specs = EXTENT_AXES_METADATA.get(tag or "", [])
    if not specs:
        return True

    is_fraction = tag in FRACTION_EXTENT_TAGS

    if indices is not None:
        count = len(indices)
    elif tag == "time_uniform":
        count = N if N is not None else 0
    else:
        first_kind = specs[0]["kind"]
        first_field = (
            specs[0]["field"]
            if first_kind == "point"
            else (
                specs[0]["origin"]
                if first_kind == "origin_span"
                else specs[0]["min_field"]
            )
        )
        count = len(getattr(extent, first_field, []))

    # Empty extent handling
    if count == 0:
        for b_min, b_max in bounds.values():
            if b_min is not None and b_max is not None and b_min < b_max:
                return False
        return True

    if tag == "time_uniform":
        start_time = getattr(extent, "start_time", 0.0)
        sampling_period = getattr(extent, "sampling_period", 1.0)
        b_min, b_max = bounds.get("time", (None, None))
        first_t = start_time
        last_t = start_time + (count - 1) * sampling_period
        if b_min is None:
            b_min = first_t
        if b_max is None:
            b_max = last_t
        return bool(first_t <= b_min and last_t >= b_max)

    # 1D Interval extents
    if len(specs) == 1 and specs[0]["kind"] == "origin_span":
        dim = specs[0]["dim"]
        b_min, b_max = bounds.get(dim, (None, None))
        raw_starts = getattr(extent, specs[0]["origin"], [])
        raw_spans = getattr(extent, specs[0]["span"], [])
        if hasattr(raw_starts, "to_pylist"):
            raw_starts = raw_starts.to_pylist()
        if hasattr(raw_spans, "to_pylist"):
            raw_spans = raw_spans.to_pylist()
        if indices is not None:
            raw_starts = [raw_starts[i] for i in indices]
            raw_spans = [raw_spans[i] for i in indices]
        starts = _normalize_coords(raw_starts, is_fraction)
        spans = _normalize_coords(raw_spans, is_fraction)
        pairs = sorted(zip(starts, spans))

        if b_min is None:
            b_min = pairs[0][0]
        if b_max is None:
            b_max = max(s + sp for s, sp in pairs)
        if b_min > b_max:
            return False
        if b_min == b_max:
            return True

        if pairs[0][0] > b_min:
            return False

        covered_until = b_min
        for s, sp in pairs:
            if is_fraction or isinstance(s, int):
                if s > covered_until:
                    return False
            else:
                if s > covered_until and not math.isclose(s, covered_until, rel_tol=1e-5, abs_tol=1e-5):
                    return False
            covered_until = max(covered_until, s + sp)
            if is_fraction or isinstance(covered_until, int):
                if covered_until >= b_max:
                    break
            else:
                if covered_until >= b_max or math.isclose(covered_until, b_max, rel_tol=1e-5, abs_tol=1e-5):
                    break
        if is_fraction or isinstance(covered_until, int):
            return bool(covered_until >= b_max)
        return bool(covered_until >= b_max or math.isclose(covered_until, b_max, rel_tol=1e-5, abs_tol=1e-5))

    # Discrete point extent (text)
    if tag == "text":
        b_min, b_max = bounds.get("offset", (None, None))
        raw_vals = getattr(extent, "offset", [])
        if hasattr(raw_vals, "to_pylist"):
            raw_vals = raw_vals.to_pylist()
        if indices is not None:
            raw_vals = [raw_vals[i] for i in indices]
        if b_min is None:
            b_min = min(raw_vals)
        if b_max is None:
            b_max = max(raw_vals)
        if b_min > b_max:
            return False
        if b_min == b_max:
            return b_min in raw_vals
        observed = set(raw_vals)
        return all(k in observed for k in range(int(b_min), int(b_max) + 1))

    # 2D Box extents (pixel_box, time_frequency_box)
    if tag in ("pixel_box", "time_frequency_box"):
        if tag == "pixel_box":
            b_x = bounds.get("x", (None, None))
            b_y = bounds.get("y", (None, None))
            xs = getattr(extent, "x", [])
            ws = getattr(extent, "width", [])
            ys = getattr(extent, "y", [])
            hs = getattr(extent, "height", [])
            if hasattr(xs, "to_pylist"):
                xs = xs.to_pylist()
            if hasattr(ws, "to_pylist"):
                ws = ws.to_pylist()
            if hasattr(ys, "to_pylist"):
                ys = ys.to_pylist()
            if hasattr(hs, "to_pylist"):
                hs = hs.to_pylist()
            if indices is not None:
                xs = [xs[i] for i in indices]
                ws = [ws[i] for i in indices]
                ys = [ys[i] for i in indices]
                hs = [hs[i] for i in indices]
            boxes = [(x, x + w, y, y + h) for x, w, y, h in zip(xs, ws, ys, hs)]
        else:
            b_x = bounds.get("time", (None, None))
            b_y = bounds.get("frequency", (None, None))
            ts = getattr(extent, "time", [])
            ds = getattr(extent, "duration", [])
            f_mins = getattr(extent, "freq_min", [])
            f_maxs = getattr(extent, "freq_max", [])
            if hasattr(ts, "to_pylist"):
                ts = ts.to_pylist()
            if hasattr(ds, "to_pylist"):
                ds = ds.to_pylist()
            if hasattr(f_mins, "to_pylist"):
                f_mins = f_mins.to_pylist()
            if hasattr(f_maxs, "to_pylist"):
                f_maxs = f_maxs.to_pylist()
            if indices is not None:
                ts = [ts[i] for i in indices]
                ds = [ds[i] for i in indices]
                f_mins = [f_mins[i] for i in indices]
                f_maxs = [f_maxs[i] for i in indices]
            boxes = [
                (t, t + d, f_min, f_max)
                for t, d, f_min, f_max in zip(ts, ds, f_mins, f_maxs)
            ]

        x_min = b_x[0] if b_x[0] is not None else min(b[0] for b in boxes)
        x_max = b_x[1] if b_x[1] is not None else max(b[1] for b in boxes)
        y_min = b_y[0] if b_y[0] is not None else min(b[2] for b in boxes)
        y_max = b_y[1] if b_y[1] is not None else max(b[3] for b in boxes)

        if x_min >= x_max or y_min >= y_max:
            return bool(x_min == x_max and y_min == y_max)

        target_area = float((x_max - x_min) * (y_max - y_min))
        float_boxes = [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in boxes
        ]
        covered_area = _union_area_2d(
            float_boxes,
            (float(x_min), float(x_max), float(y_min), float(y_max)),
        )
        return abs(covered_area - target_area) < 1e-6

    # Spatiotemporal intervals
    if tag in ("cartesian_time_interval", "spherical_time_interval"):
        b_time = bounds.get("time", (None, None))
        ts = getattr(extent, "time", [])
        ds = getattr(extent, "duration", [])
        if hasattr(ts, "to_pylist"):
            ts = ts.to_pylist()
        if hasattr(ds, "to_pylist"):
            ds = ds.to_pylist()
        if indices is not None:
            ts = [ts[i] for i in indices]
            ds = [ds[i] for i in indices]
        pairs = sorted(zip(ts, ds))
        b_min = b_time[0] if b_time[0] is not None else pairs[0][0]
        b_max = b_time[1] if b_time[1] is not None else max(s + sp for s, sp in pairs)
        if b_min > b_max:
            return False
        if b_min == b_max:
            return True
        if pairs[0][0] > b_min:
            return False
        covered_until = b_min
        for s, sp in pairs:
            if is_fraction or isinstance(s, int):
                if s > covered_until:
                    return False
            else:
                if s > covered_until and not math.isclose(s, covered_until, rel_tol=1e-5, abs_tol=1e-5):
                    return False
            covered_until = max(covered_until, s + sp)
            if is_fraction or isinstance(covered_until, int):
                if covered_until >= b_max:
                    break
            else:
                if covered_until >= b_max or math.isclose(covered_until, b_max, rel_tol=1e-5, abs_tol=1e-5):
                    break
        if is_fraction or isinstance(covered_until, int):
            return bool(covered_until >= b_max)
        return bool(covered_until >= b_max or math.isclose(covered_until, b_max, rel_tol=1e-5, abs_tol=1e-5))

    # Continuous point extents
    for spec in specs:
        dim = spec["dim"]
        b_min, b_max = bounds.get(dim, (None, None))
        raw_vals = getattr(extent, spec["field"], [])
        if hasattr(raw_vals, "to_pylist"):
            raw_vals = raw_vals.to_pylist()
        if indices is not None:
            raw_vals = [raw_vals[i] for i in indices]
        vals = _normalize_coords(raw_vals, is_fraction)
        v_min = min(vals)
        v_max = max(vals)
        if b_min is None:
            b_min = v_min
        if b_max is None:
            b_max = v_max
        if b_min > b_max:
            return False
        if b_min == b_max:
            continue
        if v_min > b_min or v_max < b_max:
            return False

    return True


def is_exhaustive(
    extent_or_annotation: Any,
    *,
    bounds: dict[str, tuple[Any, Any]] | None = None,
    grouping: Any = None,
    group_id: int | None = None,
    N: int | None = None,
) -> bool:
    """
    Check whether observations in an extent or grouping fully cover the space without gaps.

    Parameters
    ----------
    extent_or_annotation : msgspec.Struct or BoppBase
        An extent struct or parent annotation instance.
    bounds : dict of str to tuple of (Any, Any) or None, optional
        Target bounds per dimension. If None, derived via get_extent_bounds.
    grouping : Grouping or None, optional
        Grouping instance to evaluate exhaustiveness within groups.
    group_id : int or None, optional
        Specific group ID to evaluate. If None and grouping is provided, verifies
        that every non-negative group ID (>= 0) is exhaustive against the extent bounds.
    N : int or None, optional
        Total number of observations (used for time_uniform extents).

    Returns
    -------
    bool
        True if the observations fully cover the bounded space without gaps, False otherwise.
    """
    extent, N = _resolve_extent_and_n(extent_or_annotation, N)
    if extent is None or extent is msgspec.UNSET:
        return False

    if grouping is not None:
        effective_bounds = (
            bounds if bounds is not None else get_extent_bounds(extent, N=N)
        )
        ids = getattr(grouping, "ids", None)
        if ids is None or ids is msgspec.UNSET:
            return _is_exhaustive_indices(
                extent, indices=[], bounds=effective_bounds, N=0
            )
        if hasattr(ids, "to_pylist"):
            ids = ids.to_pylist()
        if group_id is not None:
            indices = [i for i, gid in enumerate(ids) if gid == group_id]
            return _is_exhaustive_indices(
                extent, indices=indices, bounds=effective_bounds, N=len(indices)
            )
        else:
            groups = _partition_group_indices(ids)
            if not groups:
                return _is_exhaustive_indices(
                    extent, indices=[], bounds=effective_bounds, N=0
                )
            for indices in groups.values():
                if not _is_exhaustive_indices(
                    extent,
                    indices=indices,
                    bounds=effective_bounds,
                    N=len(indices),
                ):
                    return False
            return True

    effective_bounds = (
        bounds if bounds is not None else _get_explicit_extent_bounds(extent)
    )
    return _is_exhaustive_indices(extent, indices=None, bounds=effective_bounds, N=N)
