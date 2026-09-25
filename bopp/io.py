from __future__ import annotations

import ast
from pathlib import Path

import msgspec
import tomllib

from ._version import get_current_schema_version
from .base import BoppBase
from .core import BoppArgumentError
from .registries import get_registry
from .util import extract_header, from_dataframe, to_dataframe


def save_bopp_csv(ann: BoppBase, filepath: str | Path) -> None:
    """
    Write metadata as TOML frontmatter followed by tabular data to a CSV file.

    Parameters
    ----------
    ann : BoppBase
        The Annotation struct instance to export.
    filepath : str or pathlib.Path
        Target file path for the output CSV file.
    """
    df = to_dataframe(ann)
    metadata = extract_header(ann)

    # 1. Convert the metadata dictionary to a TOML string
    toml_bytes = msgspec.toml.encode(metadata)
    toml_text = toml_bytes.decode("utf-8")
    
    # 2. Prefix every line with a comment hash
    frontmatter = ["# ---\n"]
    for line in toml_text.splitlines():
        frontmatter.append(f"# {line}\n")
    frontmatter.append("# ---\n")

    with open(filepath, 'w', encoding='utf-8') as f:
        # 3. Write the header
        f.writelines(frontmatter)
        
        # 4. Hand the open file pointer to Pandas/Polars
        if hasattr(df, 'to_csv'):
            # is this a pandas dataframe?
            df.to_csv(f, index=False)
        elif hasattr(df, "write_csv"):
            # Is this a polars dataframe?
            df.write_csv(f)
        else:
            raise BoppArgumentError(f"Unknown dataframe type: {type(df)}")


def load_bopp_json(filepath: str | Path) -> BoppBase:
    """
    Read a BOPP JSON file directly into an Annotation model instance.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the BOPP JSON file to read.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """
    # msgspec operates fastest on raw bytes, so we read as "rb"
    with open(filepath, "rb") as f:
        data = f.read()

    raw_dict = msgspec.json.decode(data, type=dict)
    version = raw_dict.get("bopp_version", get_current_schema_version())

    annotation_cls = get_registry(version)["Annotation"]
    return msgspec.json.decode(data, type=annotation_cls)


def save_bopp_json(annotation: BoppBase, filepath: str | Path) -> None:
    """
    Serialize an Annotation instance directly into a JSON file.

    Parameters
    ----------
    annotation : BoppBase
        The Annotation instance to serialize.
    filepath : str or pathlib.Path
        Target file path for saving the JSON output.
    """
    # msgspec encodes structs natively without needing conversion dicts
    json_data = msgspec.json.encode(annotation)
    
    with open(filepath, "wb") as f:
        f.write(json_data)


# ==========================================
# 1. Saving (Encoding) to Msgpack
# ==========================================
def save_bopp_msgpack(annotation: BoppBase, filepath: str | Path) -> None:
    """
    Serialize an Annotation instance directly into a binary MsgPack file.

    Parameters
    ----------
    annotation : BoppBase
        The Annotation instance to serialize.
    filepath : str or pathlib.Path
        Target file path for saving the MsgPack binary output.
    """
    # msgspec encodes structs natively without needing conversion dicts
    binary_data = msgspec.msgpack.encode(annotation)
    
    with open(filepath, "wb") as f:
        f.write(binary_data)


# ==========================================
# 2. Loading (Decoding) from Msgpack
# ==========================================
def load_bopp_msgpack(filepath: str | Path) -> BoppBase:
    """
    Read a binary MsgPack file and decode it into an Annotation struct.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the binary MsgPack file.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """
    with open(filepath, "rb") as f:
        binary_data = f.read()

    raw_dict = msgspec.msgpack.decode(binary_data, type=dict)
    version = raw_dict.get("bopp_version", get_current_schema_version())

    annotation_cls = get_registry(version)["Annotation"]
    return msgspec.msgpack.decode(binary_data, type=annotation_cls)


def load_bopp_csv(filepath: str | Path) -> BoppBase:
    """
    Read a BOPP CSV file directly into a validated Annotation struct.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the BOPP CSV file.

    Returns
    -------
    BoppBase
        Decoded and validated Annotation instance.
    """
    import pandas as pd

    toml_lines = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # 1. Parse TOML Frontmatter
        first_line = f.readline().strip()
        
        if first_line == "# ---":
            while True:
                line = f.readline()
                if not line or line.strip() == "# ---":
                    break
                # Strip the comment hash and leading space
                toml_lines.append(line.lstrip('#').lstrip(' '))
                
            metadata = tomllib.loads("".join(toml_lines))
        else:
            # No frontmatter found, reset the file pointer
            f.seek(0)
            metadata = {}

        # 2. Hand the open file pointer directly to Pandas
        df = pd.read_csv(f)
        
    # 3. Handle Polyphonic / List Data safely
    # If a payload contains lists (e.g., ["C", "E", "G"]), the CSV writer saves them 
    # as literal strings. We evaluate them back to actual Python lists here.
    payload_cols = [c for c in df.columns if c.startswith("payload:")]
    for col in payload_cols:
        if df[col].dtype == object and df[col].astype(str).str.startswith('[').any():
            df[col] = df[col].apply(ast.literal_eval)

    # Attach the singleton fields directly to the DataFrame attributes
    df.attrs.update(metadata)

    return from_dataframe(df)
