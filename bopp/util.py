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
    Extract all singleton fields from an Annotation for TOML serialization.

    Parallel array blocks (`extent`, `payload`, `confidence`) are omitted.

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
    >>> from bopp.models.v1.annotation import Annotation
    >>> ann = Annotation(media_id="audio:123", payload={"payload_type": "onset", "time": [0.1]})
    >>> header = extract_header(ann)
    >>> header["media_id"]
    'audio:123'
    """
    excluded_fields = {"extent", "payload", "confidence"}

    header_data = {}

    for field in msgspec.structs.fields(annotation):
        if field.name in excluded_fields:
            continue

        value = getattr(annotation, field.name)

        if value is not None and value is not msgspec.UNSET:
            header_data[field.name] = msgspec.to_builtins(value)

    return header_data


def _validate_dataframe_columns(columns: list[str], bopp_version: str) -> None:
    """Validate DataFrame column headers according to BOPP rules.

    Rules:
    1) Every column must have exactly three fields separated by : (facet:type:field).
    2) All columns belonging to the same facet must have a single type.
    3) The facet and type must exist in the registry for the given version.
    4) The field must exist for the type struct.
    """
    registry = get_registry(bopp_version)
    facet_types: dict[str, str] = {}

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


def _extract_facet_data(
    struct: msgspec.Struct | None,
    facet_name: str,
    target_dict: dict[str, Any],
) -> None:
    """Extract array fields from a facet struct and add them to target_dict with formatted keys."""
    if struct is None or struct is msgspec.UNSET:
        return

    tag = _get_tag(struct)
    type_field_name = f"{facet_name}_type"

    for field in msgspec.structs.fields(type(struct)):
        if field.name == type_field_name:
            continue
        val = getattr(struct, field.name)
        target_dict[f"{facet_name}:{tag}:{field.name}"] = val


def to_dataframe(
    annotation: BoppBase,
    backend: Literal["pandas", "pandas-pyarrow", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """
    Convert an Annotation instance into a Pandas or Polars DataFrame.

    Singleton metadata is preserved in `df.attrs` (for Pandas) or custom attributes (for Polars).

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

    _extract_facet_data(getattr(annotation, "extent", msgspec.UNSET), "extent", data)
    _extract_facet_data(getattr(annotation, "payload", msgspec.UNSET), "payload", data)
    _extract_facet_data(getattr(annotation, "confidence", msgspec.UNSET), "confidence", data)

    attrs = {
        "bopp_version": getattr(annotation, "bopp_version", get_current_schema_version()),
        "media_id": getattr(annotation, "media_id", None),
    }

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
    is_pandas = False
    is_polars = False

    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            is_pandas = True
    except ImportError:
        pass

    if not is_pandas:
        try:
            import polars as pl

            if isinstance(df, pl.DataFrame):
                is_polars = True
        except ImportError:
            pass

    if not (is_pandas or is_polars):
        raise BoppArgumentError(
            f"Unsupported DataFrame type: {type(df)}. Must be a pandas or polars DataFrame."
        )

    attrs = getattr(df, "attrs", {})
    columns = list(df.columns)

    bopp_version = attrs.get("bopp_version", get_current_schema_version())

    _validate_dataframe_columns(columns, bopp_version)

    registry = get_registry(bopp_version)
    annotation_cls = registry["Annotation"]

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

    coord_cols = [c for c in columns if c.startswith("extent:")]
    payload_cols = [c for c in columns if c.startswith("payload:")]
    conf_cols = [c for c in columns if c.startswith("confidence:")]

    def _get_column_list(column_name: str) -> list[Any]:
        if is_pandas:
            return df[column_name].tolist()
        else:
            return df[column_name].to_list()

    if coord_cols:
        ext_type = coord_cols[0].split(":")[1]
        bopp_data["extent"] = {"extent_type": ext_type}
        for col in coord_cols:
            parts = col.split(":")
            field_name = parts[2]
            bopp_data["extent"][field_name] = _get_column_list(col)

    if payload_cols:
        payload_type = payload_cols[0].split(":")[1]
        bopp_data["payload"] = {"payload_type": payload_type}
        for col in payload_cols:
            parts = col.split(":")
            field_name = parts[2]
            bopp_data["payload"][field_name] = _get_column_list(col)

    if conf_cols:
        conf_type = conf_cols[0].split(":")[1]
        bopp_data["confidence"] = {"confidence_type": conf_type}
        for col in conf_cols:
            parts = col.split(":")
            field_name = parts[2]
            bopp_data["confidence"][field_name] = _get_column_list(col)

    return msgspec.convert(bopp_data, type=annotation_cls)
