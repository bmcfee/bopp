from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import tomllib

from .core import BoppArgumentError
from .models.v1.annotation import Annotation
from .util import extract_header, from_dataframe, to_dataframe

if TYPE_CHECKING:
    from pandas import DataFrame


def to_csv(ann: Annotation, filepath: str | Path) -> None:
    """
    Write metadata as TOML frontmatter followed by tabular data to a CSV file.

    Parameters
    ----------
    ann : Annotation
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
        
        # 4. Hand the open file pointer to Pandas!
        if hasattr(df, 'to_csv'):
            # is this a pandas dataframe?
            df.to_csv(f, index=False)
        elif hasattr(df, "write_csv"):
            # Is this a polars dataframe?
            df.write_csv(f)
        else:
            raise BoppArgumentError(f"Unknown dataframe type: {type(df)}")


def load_bopp_json(filepath: str | Path) -> Annotation:
    """
    Read a BOPP JSON file directly into an Annotation model instance.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the BOPP JSON file to read.

    Returns
    -------
    Annotation
        Decoded and validated Annotation instance.
    """
    # msgspec operates fastest on raw bytes, so we read as "rb"
    with open(filepath, "rb") as f:
        data = f.read()

    return msgspec.json.decode(data, type=Annotation)


def save_bopp_json(annotation: Annotation, filepath: str | Path) -> None:
    """
    Serialize an Annotation instance directly into a JSON file.

    Parameters
    ----------
    annotation : Annotation
        The Annotation instance to serialize.
    filepath : str or pathlib.Path
        Target file path for saving the JSON output.
    """
    # msgspec encodes structs natively without needing conversion dicts
    json_data = msgspec.json.encode(annotation)
    
    with open(filepath, "wb") as f:
        f.write(json_data)
    print(f"Successfully saved to {filepath} ({len(json_data)} bytes)")


# ==========================================
# 1. Saving (Encoding) to Msgpack
# ==========================================
def save_bopp_msgpack(annotation: Annotation, filepath: str | Path) -> None:
    """
    Serialize an Annotation instance directly into a binary MsgPack file.

    Parameters
    ----------
    annotation : Annotation
        The Annotation instance to serialize.
    filepath : str or pathlib.Path
        Target file path for saving the MsgPack binary output.
    """
    # msgspec encodes structs natively without needing conversion dicts
    binary_data = msgspec.msgpack.encode(annotation)
    
    with open(filepath, "wb") as f:
        f.write(binary_data)
    print(f"Successfully saved to {filepath} ({len(binary_data)} bytes)")


# ==========================================
# 2. Loading (Decoding) from Msgpack
# ==========================================
def load_bopp_msgpack(filepath: str | Path) -> Annotation:
    """
    Read a binary MsgPack file and decode it into an Annotation struct.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the binary MsgPack file.

    Returns
    -------
    Annotation
        Decoded and validated Annotation instance.
    """
    with open(filepath, "rb") as f:
        binary_data = f.read()
        
    # Decode and instantly validate against the Annotation schema
    return msgspec.msgpack.decode(binary_data, type=Annotation)


def read_bopp_csv(filepath: str | Path) -> DataFrame:
    """
    Read a BOPP CSV file, extract TOML frontmatter into `df.attrs`, and return a Pandas DataFrame.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the BOPP CSV file to read.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing tabular content, with metadata attributes
        stored in `attrs`.
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
    return df


def load_bopp_csv(filepath: str | Path) -> Annotation:
    """
    Read a BOPP CSV file directly into a validated Annotation struct.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the BOPP CSV file.

    Returns
    -------
    Annotation
        Decoded and validated Annotation instance.
    """
    df = read_bopp_csv(filepath)
    return from_dataframe(df)
