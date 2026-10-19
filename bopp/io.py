from __future__ import annotations

import csv
import tomllib
import types
import typing
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import msgspec

from ._version import get_current_schema_version
from .base import BoppBase
from .core import (
    ensure_annotation_id,
    validate_and_set_annotation_id,
    validate_annotation_id,
)
from .exceptions import BoppArgumentError, BoppIOError, BoppRegistryError
from .extensions import get_extensions
from .registries import FACET_REGISTRY_KEYS, get_registry
from .util import from_dataframe, to_dataframe


class _VersionHeader(msgspec.Struct):
    """Lightweight struct for fast schema version extraction."""

    bopp_version: str = get_current_schema_version()


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

    return val


def _build_csv_converters(header_line: str, bopp_version: str) -> dict[str, Callable[[str], Any]]:
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

    complex_registry: dict[str, dict[str, list[str]]] = registry.get("COMPLEX_FIELDS_REGISTRY", {})
    string_registry: dict[str, dict[str, list[str]]] = registry.get("STRING_FIELDS_REGISTRY", {})

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


def _strip_annotated(hint: Any) -> Any:
    """Recursively strip typing.Annotated wrappers and PEP 695 type aliases."""
    while True:
        if typing.get_origin(hint) is typing.Annotated:
            args = typing.get_args(hint)
            if not args:
                break
            hint = args[0]
            continue
        # Support Python 3.12+ PEP 695 type alias statements (typing.TypeAliasType)
        value = getattr(hint, "__value__", None)
        if value is not None and value is not hint:
            hint = value
            continue
        break
    return hint


def _unwrap_type_hint(hint: Any) -> Any:
    """
    Unwrap list, Annotated, union, and optional annotations to determine inner element type.
    """
    hint = _strip_annotated(hint)

    origin = typing.get_origin(hint)
    if origin is list:
        args = typing.get_args(hint)
        if args:
            hint = _strip_annotated(args[0])
            origin = typing.get_origin(hint)

    # Handle Union / Optional (types.UnionType in 3.10+ or typing.Union)
    if origin is typing.Union or origin is types.UnionType:
        args = tuple(
            _strip_annotated(arg)
            for arg in typing.get_args(hint)
            if arg not in (type(None), type(msgspec.UNSET))
            )
        if len(args) == 1:
            return args[0]
        if all(arg in (int, float) for arg in args):
            return float
        return hint
    return hint


def _type_to_polars_dtype(
    type_hint: Any,
    field_name: str,
    tag: str,
    tag_field: str,
    complex_registry: dict[str, dict[str, list[str]]],
    string_registry: dict[str, dict[str, list[str]]],
    pl: Any,
) -> tuple[Any, bool]:
    """
    Map a struct field type annotation to a Polars DataType and complex flag.
    """
    if field_name in complex_registry.get(tag_field, {}).get(tag, []):
        return pl.String, True

    if field_name in string_registry.get(tag_field, {}).get(tag, []):
        return pl.String, False

    elem_type = _unwrap_type_hint(type_hint)

    if elem_type is float:
        return pl.Float64, False
    if elem_type is int:
        return pl.Int64, False
    if elem_type is bool:
        return pl.Boolean, False
    if elem_type is str:
        return pl.String, False

    if isinstance(elem_type, type):
        if issubclass(elem_type, float):
            return pl.Float64, False
        if issubclass(elem_type, int):
            return pl.Int64, False
        if issubclass(elem_type, bool):
            return pl.Boolean, False
        if issubclass(elem_type, str):
            return pl.String, False

    origin = typing.get_origin(elem_type)
    if origin in (list, dict, tuple):
        return pl.String, True

    # Fallback default
    return pl.String, False


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
    try:
        import polars as pl
    except ImportError as err:
        raise BoppIOError("polars is required for building polars schema") from err

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

    complex_registry: dict[str, dict[str, list[str]]] = registry.get("COMPLEX_FIELDS_REGISTRY", {})
    string_registry: dict[str, dict[str, list[str]]] = registry.get("STRING_FIELDS_REGISTRY", {})

    # Cache type hints per struct class
    struct_type_hints: dict[type[msgspec.Struct], dict[str, Any]] = {}

    for col in columns:
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
                        try:
                            struct_type_hints[struct_cls] = typing.get_type_hints(
                                struct_cls, include_extras=True
                            )
                        except (NameError, TypeError, AttributeError):
                            struct_type_hints[struct_cls] = getattr(struct_cls, "__annotations__", {})

                    hints = struct_type_hints[struct_cls]
                    field_hint = hints.get(field)
                    dtype, is_complex = _type_to_polars_dtype(
                        field_hint, field, tag, tag_field, complex_registry, string_registry, pl
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
        validate_and_set_annotation_id(ann)
    elif generate_id and validate_id:
        if getattr(ann, "id", msgspec.UNSET) in (msgspec.UNSET, None):
            ensure_annotation_id(ann)
        else:
            validate_annotation_id(ann)
    elif not generate_id and validate_id:
        validate_annotation_id(ann)


def resolve_extensions(
    ann: BoppBase,
    *,
    allow_missing: bool = True,
) -> None:
    """
    Apply registered schema extensions to the payload of an annotation.

    Converts raw payload item mappings or primitives into their registered target types
    (typically `msgspec.Struct` models) using strict validation (`strict=True`).

    Parameters
    ----------
    ann : BoppBase
        The Annotation struct instance containing an extension payload.
    allow_missing : bool, default True
        If True, issues a warning when an extension schema identifier is not
        found in the registry. If False, raises `BoppRegistryError`.

    Raises
    ------
    BoppRegistryError
        If `allow_missing` is False and the payload schema is not registered.
    msgspec.ValidationError
        If payload values fail validation against the extension schema.

    Notes
    -----
    Deterministic IDs and Round-Trip Validation:

    BOPP annotation IDs are deterministic UUIDv5 hashes of the serialized annotation
    content. Because saving and validation convert extension types back to canonical
    representations (built-in mappings/lists/scalars), validation before or after extension
    resolution is equivalent as long as the extension type's serialization and deserialization
    are strictly lossless and well-defined.

    Enforcing `strict=True` ensures that:
    1. Many implicit type coercions (such as string to float) are rejected. Lossless
       integer-to-float promotion remains permitted, so canonical wire types are still required.
    2. Data types remain predictable across environments that possess the extension schema
       and environments that leave payload items as raw dictionaries or primitives.
    """
    payload = getattr(ann, "payload", None)
    if payload is None or payload is msgspec.UNSET:
        return

    bopp_version = getattr(ann, "bopp_version", get_current_schema_version())
    extension_cls = get_registry(bopp_version)["PAYLOAD_TYPE_REGISTRY"]["ext"]
    if not isinstance(payload, extension_cls):
        return

    ext_schema = getattr(payload, "ext_schema", None)
    extensions = get_extensions()
    if not isinstance(ext_schema, str) or ext_schema not in extensions:
        if allow_missing:
            warnings.warn(
                f"No extension registered for schema '{ext_schema}'. Skipping validation.",
                UserWarning,
                stacklevel=2,
            )
            return
        raise BoppRegistryError(
            f"Extension schema '{ext_schema}' not found in registry."
        )

    extension_type = extensions[ext_schema]
    payload.value = msgspec.convert(payload.value, list[extension_type], strict=True)  # type: ignore[valid-type]


def save_bopp_csv(
    ann: BoppBase,
    filepath: str | Path,
    *,
    validate_id: bool = True,
    generate_id: bool = True,
    backend: Literal["pandas", "pandas-pyarrow", "polars"] = "polars",
) -> None:
    """
    Write metadata as TOML frontmatter followed by tabular data to a CSV file.

    Parameters
    ----------
    ann : BoppBase
        The Annotation struct instance to export.
    filepath : str or pathlib.Path
        Target file path for the output CSV file.
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

    # Serializing to dataframe stores header information in attrs
    # We encode complex fields as JSON strings for CSV compatibility across backends
    df = to_dataframe(ann, backend=backend, encode_complex=True)
    header = getattr(df, "attrs", {})

    # 1. Convert the metadata dictionary to a TOML string
    toml_bytes = msgspec.toml.encode(header)
    toml_text = toml_bytes.decode("utf-8")

    # 2. Prefix every line with a comment hash
    frontmatter = ["# ---\n"]
    for line in toml_text.splitlines():
        frontmatter.append(f"# {line}\n")
    frontmatter.append("# ---\n")

    with open(filepath, "w", encoding="utf-8") as f:
        # 3. Write the header
        f.writelines(frontmatter)

        # 4. Hand the open file pointer to Pandas/Polars
        if hasattr(df, "to_csv"):
            # is this a pandas dataframe?
            df.to_csv(f, index=False)
        elif hasattr(df, "write_csv"):
            # Is this a polars dataframe?
            df.write_csv(f, quote_style="non_numeric")
        else:
            raise BoppArgumentError(f"Unknown dataframe type: {type(df)}")


def _save_msgspec_encoded(
    annotation: BoppBase,
    filepath: str | Path,
    codec: Any,
    *,
    validate_id: bool,
    generate_id: bool,
) -> None:
    _prepare_annotation_for_save(annotation, validate_id=validate_id, generate_id=generate_id)
    encoded_data = codec.encode(annotation)
    with open(filepath, "wb") as f:
        f.write(encoded_data)


def _load_msgspec_encoded(
    filepath: str | Path,
    codec: Any,
    *,
    validate_id: bool,
) -> BoppBase:
    with open(filepath, "rb") as f:
        data = f.read()

    header = codec.decode(data, type=_VersionHeader)
    annotation_cls = get_registry(header.bopp_version)["Annotation"]
    ann = codec.decode(data, type=annotation_cls)

    if validate_id:
        validate_annotation_id(ann)

    return ann


def load_bopp_json(filepath: str | Path, *, validate_id: bool = True, resolve_ext: bool = True) -> BoppBase:
    """
    Read a BOPP JSON file directly into an Annotation model instance.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the BOPP JSON file to read.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID after loading.
    resolve_ext : bool, default True
        If True, resolves and validates extension payload values via registered extensions.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """

    ann = _load_msgspec_encoded(filepath, msgspec.json, validate_id=validate_id)
    if resolve_ext:
        resolve_extensions(ann)
    return ann

def save_bopp_json(
    annotation: BoppBase,
    filepath: str | Path,
    *,
    validate_id: bool = True,
    generate_id: bool = True,
) -> None:
    """
    Serialize an Annotation instance directly into a JSON file.

    Parameters
    ----------
    annotation : BoppBase
        The Annotation instance to serialize.
    filepath : str or pathlib.Path
        Target file path for saving the JSON output.
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
        filepath,
        msgspec.json,
        validate_id=validate_id,
        generate_id=generate_id,
    )


def save_bopp_msgpack(
    annotation: BoppBase,
    filepath: str | Path,
    *,
    validate_id: bool = True,
    generate_id: bool = True,
) -> None:
    """
    Serialize an Annotation instance directly into a binary MsgPack file.

    Parameters
    ----------
    annotation : BoppBase
        The Annotation instance to serialize.
    filepath : str or pathlib.Path
        Target file path for saving the MsgPack binary output.
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
        filepath,
        msgspec.msgpack,
        validate_id=validate_id,
        generate_id=generate_id,
    )


def load_bopp_msgpack(
    filepath: str | Path,
    *,
    validate_id: bool = True,
    resolve_ext: bool = True,
) -> BoppBase:
    """
    Read a binary MsgPack file and decode it into an Annotation struct.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the binary MsgPack file.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID after loading.
    resolve_ext : bool, default True
        If True, resolves and validates extension payload values via registered extensions.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """
    ann = _load_msgspec_encoded(filepath, msgspec.msgpack, validate_id=validate_id)
    if resolve_ext:
        resolve_extensions(ann)

    return ann


def load_bopp_csv(
    filepath: str | Path,
    *,
    validate_id: bool = True,
    resolve_ext: bool = True,
    backend: Literal["polars", "pandas"] = "polars",
) -> BoppBase:
    """
    Read a BOPP CSV file directly into a validated Annotation struct.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the BOPP CSV file.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID after loading.
    resolve_ext : bool, default True
        If True, resolves and validates extension payload values via registered extensions.
        Emits a warning if missing.
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

    with open(filepath, "r", encoding="utf-8") as f:
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
            # No frontmatter found, reset the file pointer
            f.seek(0)
            metadata = {}

        bopp_version = metadata.get("bopp_version", get_current_schema_version())
        header_pos = f.tell()
        header_line = f.readline()
        f.seek(header_pos)

        # 2. Hand the open file pointer to Polars or Pandas
        if backend == "polars":
            try:
                import polars as pl
            except ImportError as err:
                raise BoppIOError("polars is required for backend='polars'") from err

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

            converters = _build_csv_converters(header_line, bopp_version)
            df = pd.read_csv(f, converters=converters if converters else None)  # type: ignore[assignment]
        else:
            raise BoppArgumentError(f"Unsupported backend: {backend}")

    # Attach the singleton fields directly to the DataFrame attributes
    if hasattr(df, "attrs"):
        df.attrs.update(metadata)
    else:
        df.attrs = metadata  # type: ignore[attr-defined]

    ann = from_dataframe(df)

    if resolve_ext:
        resolve_extensions(ann)

    if validate_id:
        validate_annotation_id(ann)

    return ann
