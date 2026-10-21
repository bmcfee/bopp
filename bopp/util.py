from __future__ import annotations

import types
import typing
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, get_args, get_origin

import msgspec

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
    import polars as pl

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


def extract_header(annotation: BoppBase) -> dict[str, Any]:
    """
    Extract all singleton/scalar fields from an Annotation for TOML serialization.

    Parallel array blocks within facets (`extent`, `payload`, `confidence`) are omitted,
    while scalar attributes on those facets are preserved under their facet key.

    Parameters
    ----------
    annotation : BoppBase
        The Annotation struct to extract metadata fields from.

    Returns
    -------
    dict of str to Any
        Dictionary of serialized header/singleton fields.

    See Also
    --------
    to_dataframe : Extracts parallel arrays and header into an AnnotationDataFrame.

    Examples
    --------
    >>> import bopp
    >>> ann = bopp.create(media_id="audio:123", payload_kind="onset", value=[0.1])
    >>> header = extract_header(ann)
    >>> header["media_id"]
    'audio:123'
    """
    header_data: dict[str, Any] = {}

    for field in msgspec.structs.fields(annotation):
        value = getattr(annotation, field.name)

        if value is None or value is msgspec.UNSET:
            continue

        if field.name in FACET_REGISTRY_KEYS:
            if isinstance(value, msgspec.Struct):
                facet_attrs: dict[str, Any] = {}
                _extract_facet_data(value, field.name, {}, facet_attrs)
                if field.name in facet_attrs:
                    tag = _get_tag(value)
                    facet_dict = {f"{field.name}_type": tag} if tag else {}
                    facet_dict.update(facet_attrs[field.name])
                    header_data[field.name] = facet_dict
        else:
            header_data[field.name] = msgspec.to_builtins(value)

    return header_data


def _validate_dataframe_columns(
    columns: list[str], registry_or_version: dict[str, Any] | str
) -> dict[str, list[tuple[str, str, str]]]:
    """
    Validate DataFrame column headers against the schema and group them by facet.

    Enforces the following validation rules:
    1. Every column must follow the format ``"facet:type:field"`` with exactly
       three colon-separated components.
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
    dict of str to list of tuple of (str, str, str)
        Mapping from facet name (`"extent"`, `"payload"`, `"confidence"`) to
        lists of ``(type_tag, field_name, full_col_name)`` tuples.

    Raises
    ------
    BoppValidationError
        If any of the four structural or schema validation rules are violated.
    """
    if isinstance(registry_or_version, str):
        registry = get_registry(registry_or_version)
    else:
        registry = registry_or_version

    facet_types: dict[str, str] = {}
    grouped_cols: dict[str, list[tuple[str, str, str]]] = {k: [] for k in FACET_REGISTRY_KEYS}

    for col in columns:
        parts = col.split(":")
        # Rule 1: Must have exactly 3 fields (facet:type:field)
        if len(parts) != 3:
            raise BoppValidationError(
                f"Invalid column format '{col}'. Expected exactly 3 colon-separated fields 'facet:type:field'."
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
        valid_fields = {f.name for f in msgspec.structs.fields(struct_cls)}
        if field_name not in valid_fields:
            raise BoppValidationError(
                f"Field '{field_name}' in column '{col}' does not exist on type '{type_tag}'."
            )

        grouped_cols[facet].append((type_tag, field_name, col))

    return grouped_cols


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
    Return the item type ``T`` of a columnar field ``list[T]``.

    Optional columns (``list[T] | UnsetType``, ``list[T] | None``) are accepted.
    Returns None if the field is not a list column.
    """
    arms = _union_arms(field_type)
    if len(arms) != 1 or get_origin(arms[0]) is not list:
        return None
    args = get_args(arms[0])
    return args[0] if args else Any


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
    Map a columnar struct field type (e.g. ``list[float]``) to a Polars DataType.

    Shared by DataFrame construction and CSV reading, so that a column is written and
    read back with the same dtype. Returns None for non-list fields and for items
    without a single scalar dtype.
    """
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

    for field in msgspec.structs.fields(type(struct)):
        if field.name == type_field_name:
            continue
        val = getattr(struct, field.name)
        if val is None or val is msgspec.UNSET:
            continue

        if isinstance(val, list):
            col_key = f"{facet_name}:{tag}:{field.name}"
            is_complex = False
            if encode_complex:
                is_complex = (
                    complex_fields is not None and field.name in complex_fields
                ) or any(
                    isinstance(item, (list, dict, tuple, msgspec.Struct))
                    for item in val
                    if item is not None
                )
                if is_complex:
                    target_dict[col_key] = [
                        msgspec.json.encode(item).decode("utf-8")
                        if item is not None and item is not msgspec.UNSET
                        else None
                        for item in val
                    ]
                else:
                    target_dict[col_key] = val
            else:
                target_dict[col_key] = val

            if column_types is not None:
                column_types[col_key] = (field.type, is_complex)
        else:
            facet_attrs[field.name] = msgspec.to_builtins(val)

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
    extract_header : Extract singleton header fields from an Annotation.

    Examples
    --------
    Direct unpacking into dataframe and header objects:

    >>> from bopp.models.v1.annotation import Annotation
    >>> ann = Annotation(
    ...     media_id="audio:123",
    ...     payload={"payload_type": "onset", "value": [1, 2]}
    ... )
    >>> df, header = to_dataframe(ann, backend="pandas")
    >>> header["media_id"]
    'audio:123'
    >>> df.columns.tolist()
    ['payload:onset:value']

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

    def _get_complex_fields_for_facet(struct: msgspec.Struct | None, facet_name: str) -> set[str]:
        if struct is None or struct is msgspec.UNSET:
            return set()
        tag = _get_tag(struct)
        if not tag:
            return set()
        tag_key = f"{facet_name}_type"
        return set(complex_registry.get(tag_key, {}).get(tag, []))

    column_types: dict[str, tuple[Any, bool]] = {}

    payload_struct = getattr(annotation, "payload", None)
    _extract_facet_data(
        payload_struct,
        "payload",
        data,
        header,
        encode_complex=encode_complex,
        complex_fields=_get_complex_fields_for_facet(payload_struct, "payload"),
        column_types=column_types,
    )

    if hasattr(annotation, "extent"):
        extent_struct = getattr(annotation, "extent", None)
        _extract_facet_data(
            extent_struct,
            "extent",
            data,
            header,
            encode_complex=encode_complex,
            complex_fields=_get_complex_fields_for_facet(extent_struct, "extent"),
            column_types=column_types,
        )
    if hasattr(annotation, "confidence"):
        confidence_struct = getattr(annotation, "confidence", None)
        _extract_facet_data(
            confidence_struct,
            "confidence",
            data,
            header,
            encode_complex=encode_complex,
            complex_fields=_get_complex_fields_for_facet(confidence_struct, "confidence"),
            column_types=column_types,
        )

    for field in msgspec.structs.fields(annotation):
        if field.name in FACET_REGISTRY_KEYS or field.name in ("bopp_version", "media_id"):
            continue
        val = getattr(annotation, field.name, msgspec.UNSET)
        if val not in (msgspec.UNSET, None):
            header[field.name] = msgspec.to_builtins(val)

    if backend in ("pandas", "pandas-pyarrow"):
        try:
            import pandas as pd
        except ImportError as err:
            raise BoppIOError("pandas is required for backend='pandas'") from err

        if backend == "pandas-pyarrow":
            try:
                import pyarrow  # noqa: F401
            except ImportError as err:
                raise BoppIOError("pyarrow is required for backend='pandas-pyarrow'") from err

            df_pd = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
        else:
            df_pd = pd.DataFrame(data)

        return AnnotationDataFrame(df=df_pd, header=header)

    elif backend == "polars":
        import polars as pl

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
    try:
        import polars as pl
        valid = isinstance(df, pl.DataFrame)
    except ImportError:
        valid = False

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

    grouped_cols = _validate_dataframe_columns(columns, registry)

    media_id = header.get("media_id")
    if not media_id or not isinstance(media_id, str):
        raise BoppValidationError(
            "Missing or invalid required 'media_id' in DataFrame header attributes."
        )

    bopp_data: dict[str, Any] = {
        "bopp_version": bopp_version,
        "media_id": media_id,
    }

    for key in ("metadata", "sandbox", "annotated_domain", "id", "parents"):
        if key in header:
            bopp_data[key] = header[key]

    def _get_column_list(col_name: str) -> list[Any]:
        col_series = df[col_name]
        extractor = getattr(col_series, "to_list", getattr(col_series, "tolist", None))
        return extractor() if extractor is not None else list(col_series)

    for facet, cols in grouped_cols.items():
        if not cols:
            continue

        facet_type = cols[0][0]
        facet_dict: dict[str, Any] = {f"{facet}_type": facet_type}

        if isinstance(header.get(facet), dict):
            facet_dict.update(header[facet])

        for _, field_name, full_col_name in cols:
            facet_dict[field_name] = _get_column_list(full_col_name)

        bopp_data[facet] = facet_dict

    return msgspec.convert(bopp_data, type=annotation_cls)
