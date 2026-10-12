from __future__ import annotations

import ast
import csv
import tomllib
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args, get_origin

import msgspec

from ._version import get_current_schema_version
from .base import BoppBase
from .core import (
    ensure_annotation_id,
    validate_and_set_annotation_id,
    validate_annotation_id,
)
from .exceptions import BoppArgumentError
from .registries import get_registry
from .util import FACET_REGISTRY_KEYS, from_dataframe, to_dataframe


class _VersionHeader(msgspec.Struct):
    """Lightweight struct for fast schema version extraction."""

    bopp_version: str = get_current_schema_version()


def _unwrap_type(tp: Any) -> Any:
    """Unwrap TypeAliasType and Annotated to inspect the underlying type."""
    while True:
        if isinstance(tp, typing.TypeAliasType):
            tp = tp.__value__
        elif hasattr(tp, "__metadata__"):
            args = get_args(tp)
            tp = args[0] if args else tp
        else:
            break
    return tp


def _is_complex_type(tp: Any) -> bool:
    """Check if an unwrapped element type is a container or complex structure."""
    tp = _unwrap_type(tp)
    if tp is Any:
        return True

    origin = get_origin(tp)
    if origin in (list, dict, tuple, set, getattr(__builtins__, "frozenset", set)):
        return True

    if origin is getattr(type(int | str), "__origin__", None) or origin is Any:
        args = get_args(tp)
        return any(_is_complex_type(arg) for arg in args if arg is not type(None))

    if isinstance(tp, type):
        return not issubclass(tp, (int, float, str, bool, bytes))

    tp_str = str(tp)
    return any(k in tp_str for k in ("Fraction", "list", "dict", "tuple", "Any"))


def _get_column_converter(type_hint: Any) -> Callable[[str], Any] | None:
    """Infer a pandas cell converter function based on struct field type hint."""
    unwrapped = _unwrap_type(type_hint)
    origin = get_origin(unwrapped)

    inner_type = unwrapped
    if origin in (list, typing.Sequence):
        args = get_args(unwrapped)
        if args:
            inner_type = _unwrap_type(args[0])

    # If the inner element is complex (e.g. fraction tuple, list, dict), parse via ast.literal_eval
    if _is_complex_type(inner_type):
        def parse_complex(val: str) -> Any:
            if not isinstance(val, str) or not val.strip():
                return None
            val_stripped = val.strip()
            if val_stripped.startswith(("[", "(", "{")):
                return ast.literal_eval(val_stripped)
            return val
        return parse_complex

    # If the inner element is explicitly a string, ensure pandas does not cast numeric-like strings to numbers
    if isinstance(inner_type, type) and issubclass(inner_type, str):
        def parse_str(val: str) -> Any:
            if val == "":
                return ""
            return str(val)
        return parse_str

    return None


def _build_csv_converters(header_line: str, bopp_version: str) -> dict[str, Callable[[str], Any]]:
    """Inspect CSV header line and generate a converters dictionary for pd.read_csv."""
    converters: dict[str, Callable[[str], Any]] = {}
    if not header_line:
        return converters

    reader = csv.reader([header_line.strip()])
    columns = next(reader, [])

    try:
        registry = get_registry(bopp_version)
    except Exception:
        return converters

    for col in columns:
        parts = col.split(":")
        if len(parts) == 3:
            facet, tag, field = parts[0], parts[1], parts[2]
            if facet in FACET_REGISTRY_KEYS:
                type_registry = registry.get(FACET_REGISTRY_KEYS[facet], {})
                struct_cls = type_registry.get(tag)
                if struct_cls is not None:
                    try:
                        hints = typing.get_type_hints(struct_cls, include_extras=True)
                        if field in hints:
                            converter = _get_column_converter(hints[field])
                            if converter is not None:
                                converters[col] = converter
                    except Exception:
                        pass
    return converters


def _prepare_annotation_for_save(
    ann: BoppBase, *, validate_id: bool, generate_id: bool
) -> None:
    """
    Helper to generate or validate the annotation ID prior to serialization.

    The four combinations of (generate_id, validate_id) operate as follows:
    - generate_id=False, validate_id=False: Do nothing.
    - generate_id=False, validate_id=True: Validate existing ID.
    - generate_id=True, validate_id=False: Force generate / overwrite existing ID.
    - generate_id=True, validate_id=True: Generate if missing, otherwise validate existing ID.
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


def save_bopp_csv(
    ann: BoppBase,
    filepath: str | Path,
    *,
    validate_id: bool = True,
    generate_id: bool = True,
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
    """
    _prepare_annotation_for_save(ann, validate_id=validate_id, generate_id=generate_id)

    # Serializing to dataframe stores header information in the attrs
    df = to_dataframe(ann)
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
            df.write_csv(f)
        else:
            raise BoppArgumentError(f"Unknown dataframe type: {type(df)}")


def load_bopp_json(filepath: str | Path, *, validate_id: bool = True) -> BoppBase:
    """
    Read a BOPP JSON file directly into an Annotation model instance.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the BOPP JSON file to read.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID after loading.
        Emits a warning if missing.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """
    # msgspec operates fastest on raw bytes, so we read as "rb"
    with open(filepath, "rb") as f:
        data = f.read()

    header = msgspec.json.decode(data, type=_VersionHeader)
    annotation_cls = get_registry(header.bopp_version)["Annotation"]
    ann = msgspec.json.decode(data, type=annotation_cls)

    if validate_id:
        validate_annotation_id(ann)

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
    _prepare_annotation_for_save(annotation, validate_id=validate_id, generate_id=generate_id)

    # msgspec encodes structs natively without needing conversion dicts
    json_data = msgspec.json.encode(annotation)

    with open(filepath, "wb") as f:
        f.write(json_data)


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
    _prepare_annotation_for_save(annotation, validate_id=validate_id, generate_id=generate_id)

    binary_data = msgspec.msgpack.encode(annotation)

    with open(filepath, "wb") as f:
        f.write(binary_data)


def load_bopp_msgpack(filepath: str | Path, *, validate_id: bool = True) -> BoppBase:
    """
    Read a binary MsgPack file and decode it into an Annotation struct.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the binary MsgPack file.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID after loading.
        Emits a warning if missing.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """
    with open(filepath, "rb") as f:
        binary_data = f.read()

    header = msgspec.msgpack.decode(binary_data, type=_VersionHeader)
    annotation_cls = get_registry(header.bopp_version)["Annotation"]
    ann = msgspec.msgpack.decode(binary_data, type=annotation_cls)

    if validate_id:
        validate_annotation_id(ann)

    return ann


def load_bopp_csv(filepath: str | Path, *, validate_id: bool = True) -> BoppBase:
    """
    Read a BOPP CSV file directly into a validated Annotation struct.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the BOPP CSV file.
    validate_id : bool, default True
        If True, validates the deterministic UUIDv5 ID after loading.
        Emits a warning if missing.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """
    import pandas as pd

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

        # 2. Inspect CSV header to infer column converters based on schema registry
        bopp_version = metadata.get("bopp_version", get_current_schema_version())
        header_pos = f.tell()
        header_line = f.readline()
        f.seek(header_pos)

        converters = _build_csv_converters(header_line, bopp_version)

        # 3. Hand the open file pointer directly to Pandas with inferred converters
        df = pd.read_csv(f, converters=converters if converters else None)

    # Attach the singleton fields directly to the DataFrame attributes
    df.attrs.update(metadata)

    ann = from_dataframe(df)

    if validate_id:
        validate_annotation_id(ann)

    return ann
