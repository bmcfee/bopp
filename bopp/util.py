from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import msgspec

from .core import BoppArgumentError, BoppIOError
from .models.v1.annotation import Annotation

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

# ==========================================
# 1. Struct <-> DataFrame Translators
# ==========================================

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
    >>> payload = BeatPositionPayload(position=[1, 2], beat=[1.0, 2.0])
    >>> _get_tag(payload)
    'beat'
    """
    config = getattr(struct, "__struct_config__", None)
    if config is not None:
        return getattr(config, "tag", None)
    return None


def extract_header(annotation: Annotation) -> dict[str, Any]:
    """
    Extract all singleton fields from an Annotation for TOML serialization.

    Parallel array blocks (`extent`, `payload`, `confidence`) are omitted.

    Parameters
    ----------
    annotation : Annotation
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


def to_dataframe(
    annotation: Annotation,
    backend: Literal["pandas", "pandas-pyarrow", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """
    Convert an Annotation instance into a Pandas or Polars DataFrame.

    Singleton metadata is preserved in `df.attrs` (for Pandas) or custom attributes (for Polars).

    Parameters
    ----------
    annotation : Annotation
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

    # 1. Parse Extents (Geometry) if present
    if annotation.extent is not None and annotation.extent is not msgspec.UNSET:
        ext_type = _get_tag(annotation.extent)
        for field in msgspec.structs.fields(type(annotation.extent)):
            if field.name == "extent_type":
                continue
            val = getattr(annotation.extent, field.name)
            data[f"extent:{ext_type}:{field.name}"] = val

    # 2. Parse Payload (Passenger Data)
    payload_type = _get_tag(annotation.payload)
    for field in msgspec.structs.fields(type(annotation.payload)):
        if field.name == "payload_type":
            continue
        val = getattr(annotation.payload, field.name)
        data[f"payload:{payload_type}:{field.name}"] = val

    # 3. Parse Confidence (if present)
    if annotation.confidence is not None and annotation.confidence is not msgspec.UNSET:
        conf_type = _get_tag(annotation.confidence)
        for field in msgspec.structs.fields(type(annotation.confidence)):
            if field.name == "confidence_type":
                continue
            val = getattr(annotation.confidence, field.name)
            data[f"confidence:{conf_type}:{field.name}"] = val

    attrs = {
        "bopp_version": annotation.bopp_version,
        "media_id": annotation.media_id,
    }
    if annotation.metadata:
        attrs["metadata"] = msgspec.to_builtins(annotation.metadata)

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


def from_dataframe(df: Any) -> Annotation:
    """
    Reconstitute a strictly typed Annotation struct from a Pandas or Polars DataFrame.

    Expects singleton fields to be present in `df.attrs` or custom attributes.

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        DataFrame with self-describing column names and attributes.

    Returns
    -------
    Annotation
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

    bopp_data: dict[str, Any] = {
        "bopp_version": attrs.get("bopp_version", "1.0.0"),
        "media_id": attrs.get("media_id", "unknown:media"),
        "payload": {},
    }

    if "metadata" in attrs:
        bopp_data["metadata"] = attrs["metadata"]
    if "annotated_domain" in attrs:
        bopp_data["annotated_domain"] = attrs["annotated_domain"]

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
            field_name = parts[2] if len(parts) > 2 else "values"
            bopp_data["extent"][field_name] = _get_column_list(col)

    if payload_cols:
        payload_type = payload_cols[0].split(":")[1]
        bopp_data["payload"]["payload_type"] = payload_type
        for col in payload_cols:
            parts = col.split(":")
            field_name = parts[2] if len(parts) > 2 else "values"
            bopp_data["payload"][field_name] = _get_column_list(col)

    if conf_cols:
        conf_type = conf_cols[0].split(":")[1]
        bopp_data["confidence"] = {"confidence_type": conf_type}
        for col in conf_cols:
            parts = col.split(":")
            field_name = parts[2] if len(parts) > 2 else "confidence"
            bopp_data["confidence"][field_name] = _get_column_list(col)

    return msgspec.convert(bopp_data, type=Annotation)
