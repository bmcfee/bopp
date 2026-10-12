from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import msgspec

from ._version import get_current_schema_version
from .base import BoppBase
from .exceptions import BoppArgumentError, BoppIOError, BoppValidationError
from .registries import get_registry

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

# ==========================================
# 1. Struct <-> DataFrame Translators
# ==========================================

FACET_REGISTRY_KEYS = {
    "extent": "EXTENT_TYPE_REGISTRY",
    "payload": "PAYLOAD_TYPE_REGISTRY",
    "confidence": "CONFIDENCE_TYPE_REGISTRY",
}


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
    to_dataframe : Extracts parallel arrays into a DataFrame and header to `df.attrs`.

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
    """Validate DataFrame column headers and return columns grouped by facet.

    Returns
    -------
    dict[str, list[tuple[str, str, str]]]
        Mapping from facet name to list of (type_tag, field_name, full_col_name).

    Rules:
    1) Every column must have exactly three fields separated by : (facet:type:field).
    2) All columns belonging to the same facet must have a single type.
    3) The facet and type must exist in the registry for the given version.
    4) The field must exist for the type struct.
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


def _extract_facet_data(
    struct: msgspec.Struct | None,
    facet_name: str,
    target_dict: dict[str, Any],
    attrs_dict: dict[str, Any],
) -> None:
    """Extract array fields from a facet struct into target_dict, and non-array scalar fields into attrs_dict."""
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
            target_dict[f"{facet_name}:{tag}:{field.name}"] = val
        else:
            facet_attrs[field.name] = msgspec.to_builtins(val)

    if facet_attrs:
        attrs_dict[facet_name] = facet_attrs


def to_dataframe(
    annotation: BoppBase,
    backend: Literal["pandas", "pandas-pyarrow", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """
    Convert an Annotation instance into a Pandas or Polars DataFrame.

    Singleton metadata and scalar facet attributes are preserved in `df.attrs` (for Pandas)
    or custom attributes (for Polars).

    Parameters
    ----------
    annotation : BoppBase
        The Annotation struct instance to convert.
    backend : {"pandas", "pandas-pyarrow", "polars"}, default "pandas"
        The DataFrame framework and backend to return.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        DataFrame representing parallel array fields with column names prefixed
        by facet and tag.

    See Also
    --------
    from_dataframe : Reconstitute an Annotation struct from a DataFrame.
    extract_header : Extract singleton header fields from an Annotation.

    Examples
    --------
    >>> from bopp.models.v1.annotation import Annotation
    >>> ann = Annotation(
    ...     media_id="audio:123",
    ...     payload={"payload_type": "onset", "time": [0.1, 0.5]}
    ... )
    >>> df = to_dataframe(ann, backend="pandas")
    >>> df.attrs["media_id"]
    'audio:123'
    """
    data: dict[str, Any] = {}

    attrs = {
        "bopp_version": getattr(annotation, "bopp_version", get_current_schema_version()),
        "media_id": getattr(annotation, "media_id", None),
    }

    _extract_facet_data(annotation.payload, "payload", data, attrs)  # type: ignore[attr-defined]
    if hasattr(annotation, "extent"):
        _extract_facet_data(annotation.extent, "extent", data, attrs)
    if hasattr(annotation, "confidence"):
        _extract_facet_data(annotation.confidence, "confidence", data, attrs)

    metadata = getattr(annotation, "metadata", None)
    if metadata is not None and metadata is not msgspec.UNSET:
        attrs["metadata"] = msgspec.to_builtins(metadata)

    sandbox = getattr(annotation, "sandbox", msgspec.UNSET)
    if sandbox not in (msgspec.UNSET, None):
        attrs["sandbox"] = msgspec.to_builtins(sandbox)

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

        for k, v in attrs.items():
            df_pd.attrs[k] = v
        return df_pd

    elif backend == "polars":
        try:
            import polars as pl
        except ImportError as err:
            raise BoppIOError("polars is required for backend='polars'") from err

        df_pl = pl.DataFrame(data)
        df_pl.attrs = attrs  # type: ignore[attr-defined]
        return df_pl

    else:
        raise BoppArgumentError(f"Unsupported backend: {backend}")


def from_dataframe(df: Any) -> BoppBase:
    """
    Reconstitute a strictly typed Annotation struct from a Pandas or Polars DataFrame.

    Expects singleton fields to be present in `df.attrs` or custom attributes.

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        DataFrame with self-describing column names and attributes.

    Returns
    -------
    BoppBase
        Validated Annotation struct built from the DataFrame data.

    See Also
    --------
    to_dataframe : Convert an Annotation instance into a Pandas or Polars DataFrame.
    """
    type_name = type(df).__name__
    module_name = getattr(type(df), "__module__", "")

    if "pandas" not in module_name and "polars" not in module_name and type_name not in ("DataFrame",):
        raise BoppArgumentError(
            f"Unsupported DataFrame type: {type(df)}. Must be a pandas or polars DataFrame."
        )

    # Confirm pandas/polars type dynamically
    try:
        import pandas as pd
        valid = isinstance(df, pd.DataFrame)
    except ImportError:
        valid = False

    if not valid:
        try:
            import polars as pl
            valid = isinstance(df, pl.DataFrame)
        except ImportError:
            pass

    if not valid:
        raise BoppArgumentError(
            f"Unsupported DataFrame type: {type(df)}. Must be a pandas or polars DataFrame."
        )

    attrs = getattr(df, "attrs", {})
    columns = list(df.columns)
    bopp_version = attrs.get("bopp_version", get_current_schema_version())

    registry = get_registry(bopp_version)
    annotation_cls = registry["Annotation"]

    grouped_cols = _validate_dataframe_columns(columns, registry)

    media_id = attrs.get("media_id")
    if not media_id or not isinstance(media_id, str):
        raise BoppValidationError(
            "Missing or invalid required 'media_id' in DataFrame attributes (df.attrs)."
        )

    bopp_data: dict[str, Any] = {
        "bopp_version": bopp_version,
        "media_id": media_id,
    }

    for key in ("metadata", "sandbox", "annotated_domain", "id", "parents"):
        if key in attrs:
            bopp_data[key] = attrs[key]

    def _get_column_list(col_name: str) -> list[Any]:
        col_series = df[col_name]
        extractor = getattr(col_series, "to_list", getattr(col_series, "tolist", None))
        return extractor() if extractor is not None else list(col_series)

    for facet, cols in grouped_cols.items():
        if not cols:
            continue

        facet_type = cols[0][0]
        facet_dict: dict[str, Any] = {f"{facet}_type": facet_type}

        if isinstance(attrs.get(facet), dict):
            facet_dict.update(attrs[facet])

        for _, field_name, full_col_name in cols:
            facet_dict[field_name] = _get_column_list(full_col_name)

        bopp_data[facet] = facet_dict

    return msgspec.convert(bopp_data, type=annotation_cls)
