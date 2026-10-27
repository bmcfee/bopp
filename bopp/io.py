from __future__ import annotations

import contextlib
import csv
import io
import tomllib
import typing
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO

import msgspec
import polars as pl

from ._version import get_current_schema_version
from .base import BoppBase
from .core import (
    compute_annotation_id,
    ensure_annotation_id,
    validate_annotation_id,
)
from .exceptions import BoppArgumentError, BoppIOError, BoppRegistryError
from .extensions import resolve_extensions
from .registries import FACET_REGISTRY_KEYS, get_registry
from .util import (
    AnnotationDataFrame,
    _polars_column_dtype,
    from_dataframe,
    to_dataframe,
)

__all__ = [
    "AnnotationDataFrame",
    "load_bopp_csv",
    "load_bopp_json",
    "load_bopp_msgpack",
    "resolve_extensions",
    "save_bopp_csv",
    "save_bopp_json",
    "save_bopp_msgpack",
]


class _VersionHeader(msgspec.Struct):
    """Lightweight struct for fast schema version extraction."""

    bopp_version: str = get_current_schema_version()


def _is_path_like(obj: Any) -> bool:
    """Check if an object represents a filesystem path."""
    return (
        isinstance(obj, (str, Path))
        and not hasattr(obj, "read")
        and not hasattr(obj, "write")
    )


@typing.overload
def _open_or_yield_write(
    target: str | Path | BinaryIO,
    binary: Literal[True],
) -> Iterator[BinaryIO]: ...


@typing.overload
def _open_or_yield_write(
    target: str | Path | TextIO,
    binary: Literal[False],
) -> Iterator[TextIO]: ...


@typing.overload
def _open_or_yield_write(
    target: str | Path | TextIO | BinaryIO,
    binary: bool,
) -> Iterator[TextIO | BinaryIO]: ...


@contextlib.contextmanager  # type: ignore[misc]
def _open_or_yield_write(
    target: str | Path | TextIO | BinaryIO,
    binary: bool,
) -> Iterator[TextIO | BinaryIO]:
    """
    Context manager yielding a writable file-like object.

    If target is a path (str or Path), opens it with mode 'wb' or 'w' and closes
    it upon exit. If target is an open file handle, yields it directly without closing.
    """
    if _is_path_like(target):
        mode = "wb" if binary else "w"
        kwargs = {} if binary else {"encoding": "utf-8"}
        with open(target, mode, **kwargs) as f:  # type: ignore[arg-type,call-overload]
            yield f
    else:
        yield target  # type: ignore[misc]


def _read_bytes_from_source(
    source: str | Path | bytes | bytearray | BinaryIO | TextIO,
) -> bytes:
    """Read binary data from a path, raw bytes, or an open binary/text stream."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)

    if hasattr(source, "read"):
        content = source.read()
        if isinstance(content, str):
            return content.encode("utf-8")
        return bytes(content)

    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()

    raise BoppIOError(f"Unsupported source type for binary read: {type(source)}")


@contextlib.contextmanager
def _open_or_yield_text_stream(source: str | Path | TextIO) -> Iterator[TextIO]:
    """
    Context manager yielding a text stream for CSV parsing.

    If source is an open stream (has .read), yields it directly.
    If source is a str or Path, opens it as a filesystem path in text mode.
    """
    if hasattr(source, "read"):
        yield source  # type: ignore[misc]
    elif isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as f:
            yield f
    else:
        raise BoppIOError(f"Unsupported source type for text stream: {type(source)}")


def _parse_complex_cell(val: str | None) -> Any:
    """
    Safely evaluate container literals from a CSV cell.

    Preserves None for empty/whitespace cells. If the string represents a literal
    container (list, dict), evaluates it via JSON decoding.

    Parameters
    ----------
    val : str or None
        The raw string read from a CSV cell.

    Returns
    -------
    Any
        The evaluated container or the original string.
    """
    if val is None or not isinstance(val, str) or not (val_stripped := val.strip()):
        return None

    try:
        return msgspec.json.decode(val_stripped)
    except (msgspec.DecodeError, ValueError, SyntaxError):
        return val_stripped


def _build_csv_converters(
    header_line: str, bopp_version: str
) -> dict[str, Callable[[str], Any]]:
    """
    Inspect the CSV header row and construct column converters using the schema registry.

    Uses `COMPLEX_FIELDS_REGISTRY` to attach container parsers (e.g. for fractions,
    nested lists, dicts) and `STRING_FIELDS_REGISTRY` to preserve explicit string values
    without accidental numeric type inference.

    Parameters
    ----------
    header_line : str
        The raw header line from the CSV file containing comma-separated column names.
    bopp_version : str
        Schema version to query against the schema registry.

    Returns
    -------
    dict of str to Callable[[str], Any]
        Mapping of CSV column names (`"facet:type:field"`) to converter functions
        suitable for :func:`pandas.read_csv`.
    """
    converters: dict[str, Callable[[str], Any]] = {}
    if not header_line:
        return converters

    reader = csv.reader([header_line.strip()])
    columns = next(reader, [])

    try:
        registry = get_registry(bopp_version)
    except (BoppRegistryError, ValueError):
        return converters

    complex_registry: dict[str, dict[str, list[str]]] = registry.get(
        "COMPLEX_FIELDS_REGISTRY", {}
    )
    string_registry: dict[str, dict[str, list[str]]] = registry.get(
        "STRING_FIELDS_REGISTRY", {}
    )

    for col in columns:
        parts = col.split(":")
        if len(parts) == 3:
            facet, tag, field = parts[0], parts[1], parts[2]
            if facet in FACET_REGISTRY_KEYS:
                tag_field = f"{facet}_type"
                if field in complex_registry.get(tag_field, {}).get(tag, []):
                    converters[col] = _parse_complex_cell
                elif field in string_registry.get(tag_field, {}).get(tag, []):
                    converters[col] = str
    return converters


def _type_to_polars_dtype(
    type_hint: Any,
    field_name: str,
    tag: str,
    tag_field: str,
    complex_registry: dict[str, dict[str, list[str]]],
    pl: Any,
) -> tuple[Any, bool]:
    """
    Map a struct field type annotation to a Polars DataType and complex flag.

    Scalar columns use the same mapping as DataFrame construction for CSV output
    (``bopp.util._polars_column_dtype``), so every column is read back with the dtype it
    was written with.
    """
    if field_name in complex_registry.get(tag_field, {}).get(tag, []):
        return pl.String, True

    dtype = _polars_column_dtype(type_hint, pl)
    if dtype is None:
        # Fallback default
        return pl.String, False
    return dtype, False


def _build_polars_csv_schema(
    header_line: str, bopp_version: str
) -> tuple[dict[str, Any], list[str]]:
    """
    Inspect the CSV header row and construct an exhaustive Polars schema and complex column list.

    Determines exact Polars data types upfront for every column based on the schema registry,
    allowing Polars to bypass column type inference during CSV ingestion.

    Parameters
    ----------
    header_line : str
        The raw header line from the CSV file containing comma-separated column names.
    bopp_version : str
        Schema version to query against the schema registry.

    Returns
    -------
    tuple of (dict[str, Any], list[str])
        A tuple consisting of the full schema mapping every column name to a Polars data type,
        and a list of column names designated as complex fields requiring post-read decoding.
    """
    schema: dict[str, Any] = {}
    complex_columns: list[str] = []

    if not header_line:
        return schema, complex_columns

    reader = csv.reader([header_line.strip()])
    columns = next(reader, [])

    try:
        registry = get_registry(bopp_version)
    except (BoppRegistryError, ValueError):
        for col in columns:
            schema[col] = pl.String
        return schema, complex_columns

    complex_registry: dict[str, dict[str, list[str]]] = registry.get(
        "COMPLEX_FIELDS_REGISTRY", {}
    )

    # Cache field types per struct class
    struct_type_hints: dict[type[msgspec.Struct], dict[str, Any]] = {}

    for col in columns:
        if col.startswith("group."):
            dot_parts = col.split(".")
            if (
                len(dot_parts) == 3
                and dot_parts[0] == "group"
                and dot_parts[1].isdigit()
                and dot_parts[2] == "ids"
            ):
                schema[col] = pl.Int64
                continue

        parts = col.split(":")
        if len(parts) == 3:
            facet, tag, field = parts[0], parts[1], parts[2]
            if facet in FACET_REGISTRY_KEYS:
                tag_field = f"{facet}_type"
                facet_registry_key = FACET_REGISTRY_KEYS[facet]
                facet_dict = registry.get(facet_registry_key, {})
                struct_cls = facet_dict.get(tag)

                if struct_cls is not None:
                    if struct_cls not in struct_type_hints:
                        struct_type_hints[struct_cls] = {
                            f.name: f.type for f in msgspec.structs.fields(struct_cls)
                        }

                    hints = struct_type_hints[struct_cls]
                    field_hint = hints.get(field)
                    dtype, is_complex = _type_to_polars_dtype(
                        field_hint, field, tag, tag_field, complex_registry, pl
                    )
                    schema[col] = dtype

                    if is_complex:
                        complex_columns.append(col)
                    continue

        # Fallback for unrecognized columns
        schema[col] = pl.String

    return schema, complex_columns


def _prepare_annotation_for_save(
    ann: BoppBase, *, validate_id: bool, generate_id: bool
) -> None:
    """
    Prepare an annotation's deterministic UUIDv5 identifier before serialization.

    The four combinations of (generate_id, validate_id) operate as follows:

    - ``generate_id=False, validate_id=False``:
      Do nothing.
    - ``generate_id=False, validate_id=True``:
      Strictly validate existing ID against annotation content.
    - ``generate_id=True, validate_id=False``:
      Force generate and overwrite any existing ID with a fresh computation.
    - ``generate_id=True, validate_id=True``:
      Generate if missing, otherwise validate the existing ID.

    Parameters
    ----------
    ann : BoppBase
        The Annotation struct to validate or mutate with a new identifier.
    validate_id : bool
        Whether to enforce ID validity check.
    generate_id : bool
        Whether to generate IDs when missing or unconditionally.
    """
    if generate_id and not validate_id:
        ann.id = compute_annotation_id(ann)  # type: ignore[attr-defined]
    elif generate_id and validate_id:
        ensure_annotation_id(ann)
    elif not generate_id and validate_id:
        validate_annotation_id(ann)


def save_bopp_csv(
    ann: BoppBase,
    target: str | Path | TextIO,
    *,
    validate_id: bool = True,
    generate_id: bool = True,
    backend: Literal["pandas", "pandas-pyarrow", "polars"] = "polars",
) -> None:
    """
    Write metadata as TOML frontmatter followed by tabular data to a CSV file or stream.

    Parameters
    ----------
    ann : BoppBase
        The Annotation struct instance to export.
    target : str, pathlib.Path, or typing.TextIO
        Target file path (as str or Path) or open text stream (e.g. io.StringIO)
        for saving the CSV output.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID before saving.
        When paired with generate_id=False, strictly validates the existing ID.
        When paired with generate_id=True, validates if present or generates if missing.
    generate_id : bool, default True
        If True and validate_id=True, generates an ID if missing; if validate_id=False,
        unconditionally overrides any existing ID with a freshly computed one.
        If False and validate_id=False, skips ID generation and validation.
    backend : {"pandas", "pandas-pyarrow", "polars"}, default "polars"
        The DataFrame framework and backend to use during serialization.

        .. warning:: `pandas` backends may not preserve all data types accurately,
            in particular those involving optional fields or unions with string types.
            Loading CSV files produced by the pandas backends may result in corruption
            of content-based `id` fields.
    """
    _prepare_annotation_for_save(ann, validate_id=validate_id, generate_id=generate_id)

    # Serializing to dataframe returns AnnotationDataFrame with df and header
    # We encode complex fields as JSON strings for CSV compatibility across backends
    ann_df = to_dataframe(ann, backend=backend, encode_complex=True)
    df, header = ann_df.df, ann_df.header

    # 1. Convert the metadata dictionary to a TOML string
    toml_bytes = msgspec.toml.encode(header)
    toml_text = toml_bytes.decode("utf-8")

    # 2. Prefix every line with a comment hash
    frontmatter = ["# ---\n"]
    for line in toml_text.splitlines():
        frontmatter.append(f"# {line}\n")
    frontmatter.append("# ---\n")

    with _open_or_yield_write(target, binary=False) as f:  # type: ignore[attr-defined]
        # 3. Write the header
        f.writelines(frontmatter)

        # 4. Hand the open file pointer to Pandas/Polars
        if isinstance(df, pl.DataFrame):
            df.write_csv(f, quote_style="non_numeric")
        else:
            df.to_csv(f, index=False)


def _save_msgspec_encoded(
    annotation: BoppBase,
    target: str | Path | BinaryIO | TextIO,
    codec: Any,
    *,
    validate_id: bool,
    generate_id: bool,
) -> None:
    _prepare_annotation_for_save(
        annotation, validate_id=validate_id, generate_id=generate_id
    )
    encoded_data = codec.encode(annotation)

    with _open_or_yield_write(target, binary=True) as f:  # type: ignore[attr-defined]
        try:
            f.write(encoded_data)
        except TypeError:
            # target is a TextIO stream expecting str
            f.write(encoded_data.decode("utf-8"))  # type: ignore[arg-type]


def _load_msgspec_encoded(
    source: str | Path | bytes | bytearray | BinaryIO | TextIO,
    codec: Any,
    *,
    validate_id: bool,
    resolve_ext: bool,
) -> BoppBase:
    data = _read_bytes_from_source(source)

    header = codec.decode(data, type=_VersionHeader)
    annotation_cls = get_registry(header.bopp_version)["Annotation"]
    ann = codec.decode(data, type=annotation_cls)

    if validate_id:
        validate_annotation_id(ann)

    if resolve_ext:
        resolve_extensions(ann)

    return ann


def load_bopp_json(
    source: str | Path | bytes | bytearray | BinaryIO | TextIO,
    *,
    validate_id: bool = True,
    resolve_ext: bool = True,
) -> BoppBase:
    """
    Read a BOPP JSON resource directly into an Annotation model instance.

    Parameters
    ----------
    source : str, pathlib.Path, bytes, bytearray, typing.BinaryIO, or typing.TextIO
        Path to the BOPP JSON file (as str or Path), an open binary or text stream
        (e.g., io.StringIO, io.BytesIO), or raw JSON bytes / bytearray.
        Note: string inputs (str) are treated strictly as filesystem paths; wrap
        in-memory JSON strings in `io.StringIO` or encode to `bytes`.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID after loading.
    resolve_ext : bool, default True
        If True, resolves and validates extension payload values via registered extensions.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """
    return _load_msgspec_encoded(
        source, msgspec.json, validate_id=validate_id, resolve_ext=resolve_ext
    )


def save_bopp_json(
    annotation: BoppBase,
    target: str | Path | BinaryIO | TextIO,
    *,
    validate_id: bool = True,
    generate_id: bool = True,
) -> None:
    """
    Serialize an Annotation instance directly into a JSON file or stream.

    Parameters
    ----------
    annotation : BoppBase
        The Annotation instance to serialize.
    target : str, pathlib.Path, typing.BinaryIO, or typing.TextIO
        Target file path (as str or Path) or open binary/text stream
        (e.g., io.StringIO, io.BytesIO) for saving the JSON output.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID before saving.
        When paired with generate_id=False, strictly validates the existing ID.
        When paired with generate_id=True, validates if present or generates if missing.
    generate_id : bool, default True
        If True and validate_id=True, generates an ID if missing; if validate_id=False,
        unconditionally overrides any existing ID with a freshly computed one.
        If False and validate_id=False, skips ID generation and validation.
    """
    _save_msgspec_encoded(
        annotation,
        target,
        msgspec.json,
        validate_id=validate_id,
        generate_id=generate_id,
    )


def save_bopp_msgpack(
    annotation: BoppBase,
    target: str | Path | BinaryIO,
    *,
    validate_id: bool = True,
    generate_id: bool = True,
) -> None:
    """
    Serialize an Annotation instance directly into a binary MsgPack file or stream.

    Parameters
    ----------
    annotation : BoppBase
        The Annotation instance to serialize.
    target : str, pathlib.Path, or typing.BinaryIO
        Target file path (as str or Path) or open binary stream (e.g., io.BytesIO)
        for saving the MsgPack binary output.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID before saving.
        When paired with generate_id=False, strictly validates the existing ID.
        When paired with generate_id=True, validates if present or generates if missing.
    generate_id : bool, default True
        If True and validate_id=True, generates an ID if missing; if validate_id=False,
        unconditionally overrides any existing ID with a freshly computed one.
        If False and validate_id=False, skips ID generation and validation.
    """
    _save_msgspec_encoded(
        annotation,
        target,
        msgspec.msgpack,
        validate_id=validate_id,
        generate_id=generate_id,
    )


def load_bopp_msgpack(
    source: str | Path | bytes | bytearray | BinaryIO,
    *,
    validate_id: bool = True,
    resolve_ext: bool = True,
) -> BoppBase:
    """
    Read a binary MsgPack resource and decode it into an Annotation struct.

    Parameters
    ----------
    source : str, pathlib.Path, bytes, bytearray, or typing.BinaryIO
        Path to the binary MsgPack file (as str or Path), raw msgpack bytes or bytearray,
        or an open binary stream (e.g., io.BytesIO).
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID after loading.
    resolve_ext : bool, default True
        If True, resolves and validates extension payload values via registered extensions.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """
    return _load_msgspec_encoded(
        source, msgspec.msgpack, validate_id=validate_id, resolve_ext=resolve_ext
    )


def load_bopp_csv(
    source: str | Path | TextIO,
    *,
    validate_id: bool = True,
    resolve_ext: bool = True,
    backend: Literal["polars", "pandas"] = "polars",
) -> BoppBase:
    """
    Read a BOPP CSV file or open text stream directly into a validated Annotation struct.

    Parameters
    ----------
    source : str, pathlib.Path, or typing.TextIO
        Path to the BOPP CSV file (as str or Path), or an open text stream (e.g. io.StringIO).
        Note: string inputs (str) are treated strictly as filesystem paths; wrap
        in-memory CSV strings in `io.StringIO`.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID after loading.
    resolve_ext : bool, default True
        If True, resolves and validates extension payload values via registered extensions.
    backend : {"polars", "pandas"}, default "polars"
        The DataFrame framework to use for parsing tabular CSV content.

        .. warning:: `pandas` backends may not preserve all data types accurately,
            in particular those involving optional fields or unions with string types.
            Loading CSV files produced by the pandas backends may result in corruption
            of content-based `id` fields.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """
    toml_lines = []

    with _open_or_yield_text_stream(source) as f:
        # 1. Parse TOML Frontmatter
        first_line = f.readline().strip()

        if first_line == "# ---":
            while True:
                line = f.readline()
                if not line or line.strip() == "# ---":
                    break
                # Strip the comment hash and leading space
                toml_lines.append(line.lstrip("#").lstrip(" "))

            metadata = tomllib.loads("".join(toml_lines))
        else:
            # No frontmatter found, reset the file pointer if seekable
            if hasattr(f, "seek"):
                f.seek(0)
            metadata = {}

        bopp_version = metadata.get("bopp_version", get_current_schema_version())
        header_pos = f.tell() if hasattr(f, "seek") else None
        header_line = f.readline()
        # Reset position back to header row so dataframe reader reads column names
        if hasattr(f, "seek") and header_pos is not None:
            f.seek(header_pos)
        else:
            # For non-seekable streams, recombine header with remaining lines
            remaining = header_line + f.read()
            f = io.StringIO(remaining)

        # 2. Hand the open file pointer to Polars or Pandas
        df: Any
        if backend == "polars":
            if not header_line.strip():
                df = pl.DataFrame()
            else:
                schema, complex_columns = _build_polars_csv_schema(
                    header_line, bopp_version
                )
                df = pl.read_csv(f, schema=schema if schema else None)

                if complex_columns:
                    df = df.with_columns(
                        [
                            pl.col(c).map_elements(
                                _parse_complex_cell, return_dtype=pl.Object
                            )
                            for c in complex_columns
                        ]
                    )

        elif backend == "pandas":
            try:
                import pandas as pd
            except ImportError as err:
                raise BoppIOError("pandas is required for backend='pandas'") from err

            if not header_line.strip():
                df = pd.DataFrame()
            else:
                converters = _build_csv_converters(header_line, bopp_version)
                df = pd.read_csv(f, converters=converters if converters else None)
        else:
            raise BoppArgumentError(f"Unsupported backend: {backend}")

    ann = from_dataframe(df, header=metadata)

    if resolve_ext:
        resolve_extensions(ann)

    if validate_id:
        validate_annotation_id(ann)

    return ann
