#!/usr/bin/env python

import ast
import msgspec
import pandas as pd
import pyarrow as pa
import yaml
import typing

from .util import extract_header, to_dataframe
from pathlib import Path

from bopp.models.v1.annotation import Annotation

from typing import Any, get_origin, get_args


def decode_arrow(type_hint, value):
    """
    Intercepts the msgspec parser to build native Arrow arrays 
    instead of standard Python lists.
    """
    # Loop to unwrap nested Annotated types (e.g., Annotated[Annotated[pa.Array, "float32"], Meta(...)])
    target = type_hint
    while get_origin(target) is typing.Annotated:
        target = get_args(target)[0]

    if target is pa.Array:
        # Convert the raw parsed list directly into a contiguous Arrow buffer
        return pa.array(value)
        
    raise TypeError(f"Type {type_hint} is not supported")


def encode_arrow(obj: Any) -> Any:
    """
    Intercepts PyArrow arrays during serialization and converts 
    them back to standard Python lists for Msgpack/JSON.
    """
    # Catch both contiguous Arrays and ChunkedArrays natively
    if isinstance(obj, (pa.Array, pa.ChunkedArray)):
        return obj.to_pylist()
    
    # msgspec requires you to raise a NotImplementedError if the hook 
    # receives an object it doesn't know how to handle.
    raise NotImplementedError(f"Object of type {type(obj)} is not supported")


def to_csv(ann: Annotation, filepath: str | Path) -> None:
    """Writes metadata as YAML frontmatter, followed by the DataFrame."""
    
    df = to_dataframe(ann)
    metadata = extract_header(ann)

    # 1. Convert the metadata dictionary to a YAML string
    yaml_text = yaml.dump(metadata, sort_keys=False, default_flow_style=False)
    
    # 2. Prefix every line with a comment hash
    frontmatter = ["# ---\n"]
    for line in yaml_text.splitlines():
        frontmatter.append(f"# {line}\n")
    frontmatter.append("# ---\n")

    with open(filepath, 'w', encoding='utf-8') as f:
        # 3. Write the header
        f.writelines(frontmatter)
        
        # 4. Hand the open file pointer to Pandas!
        df.to_csv(f, index=False)


def load_bopp_json(filepath: str) -> Annotation:
    """Reads a BOPP JSON file directly into the Python model."""
    # msgspec operates fastest on raw bytes, so we read as "rb"
    with open(filepath, "rb") as f:
        data = f.read()

    return msgspec.json.decode(data, type=Annotation, dec_hook=decode_arrow)


def save_bopp_json(annotation: Annotation, filepath: str) -> None:
    """Serializes a Annotation instance directly into a JSON file."""
    # msgspec encodes structs natively without needing conversion dicts
    json_data = msgspec.json.encode(annotation, enc_hook=encode_arrow)
    
    with open(filepath, "wb") as f:
        f.write(json_data)
    print(f"Successfully saved to {filepath} ({len(json_data)} bytes)")


# ==========================================
# 1. Saving (Encoding) to Msgpack
# ==========================================
def save_bopp_msgpack(annotation: Annotation, filepath: str) -> None:
    """
    Serializes a Annotation instance directly into a binary msgpack file.
    """
    # msgspec encodes structs natively without needing conversion dicts
    binary_data = msgspec.msgpack.encode(annotation, enc_hook=encode_arrow)
    
    with open(filepath, "wb") as f:
        f.write(binary_data)
    print(f"Successfully saved to {filepath} ({len(binary_data)} bytes)")


# ==========================================
# 2. Loading (Decoding) from Msgpack
# ==========================================
def load_bopp_msgpack(filepath: str) -> Annotation:
    """
    Reads a binary msgpack file and decodes/validates it back into 
    the Annotation struct (running your length checks via __post_init__).
    """
    with open(filepath, "rb") as f:
        binary_data = f.read()
        
    # Decode and instantly validate against the Annotation schema
    return msgspec.msgpack.decode(binary_data, type=Annotation, dec_hook=decode_arrow)


def read_bopp_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Reads a BOPP CSV file, extracts the YAML frontmatter into df.attrs, 
    and returns the tabular data as a Pandas DataFrame.
    """
    yaml_lines = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # 1. Parse YAML Frontmatter
        first_line = f.readline().strip()
        
        if first_line == "# ---":
            while True:
                line = f.readline()
                if not line or line.strip() == "# ---":
                    break
                # Strip the comment hash and leading space
                yaml_lines.append(line.lstrip('#').lstrip(' '))
                
            metadata = yaml.safe_load("".join(yaml_lines))
        else:
            # No frontmatter found, reset the file pointer
            f.seek(0)
            metadata = {}

        # 2. Hand the open file pointer directly to Pandas
        df = pd.read_csv(f)
        
    # 3. Handle Polyphonic / List Data safely
    # If a payload contains lists (e.g., ["C", "E", "G"]), the CSV writer saves them 
    # as literal strings. We evaluate them back to actual Python lists here.
    payload_col = next((c for c in df.columns if c.startswith("payload:")), None)
    if payload_col and df[payload_col].dtype == object and df[payload_col].str.startswith('[').any():
        df[payload_col] = df[payload_col].apply(ast.literal_eval)

    # Attach the singleton fields directly to the DataFrame attributes
    df.attrs = metadata
    return df


def from_dataframe(df: pd.DataFrame) -> Annotation:
    """
    Reconstitutes a strictly typed Annotation struct from a DataFrame.
    Expects singleton fields (like media_id, annotated_domain) to be in df.attrs.
    """
    # 1. Scaffold the base dictionary with required singletons
    bopp_data = {
        "bopp_version": df.attrs.get("bopp_version", "1.0.0"),
        "media_id": df.attrs.get("media_id", "unknown:media"),
        "extent": {"coordinates": []},
        "payload": {"values": []}
    }
    
    # Safely inject optional root fields if they exist in the YAML header
    if "metadata" in df.attrs:
        bopp_data["metadata"] = df.attrs["metadata"]
    if "annotated_domain" in df.attrs:
        bopp_data["annotated_domain"] = df.attrs["annotated_domain"]
        
    # 2. Infer structural types from the self-describing column headers
    coord_cols = [c for c in df.columns if c.startswith("extent:")]
    payload_col = next(c for c in df.columns if c.startswith("payload:"))
    conf_col = next((c for c in df.columns if c.startswith("confidence:")), None)
    
    bopp_data["extent"]["extent_type"] = coord_cols[0].split(":")[1]
    bopp_data["payload"]["payload_type"] = payload_col.split(":")[1]
    
    if conf_col:
        bopp_data["confidence"] = {
            "confidence_type": conf_col.split(":")[1],
            "confidence": df[conf_col].tolist()
        }
        
    # 3. Extract the array data natively
    bopp_data["payload"]["values"] = df[payload_col].tolist()
    
    if len(coord_cols) == 1:
        # 1D points
        bopp_data["extent"]["coordinates"] = df[coord_cols[0]].tolist()
    else:
        # N-Dimensional coordinates (e.g., time_interval [start, duration]) 
        # Stack them back into a list of lists
        bopp_data["extent"]["coordinates"] = df[coord_cols].to_numpy().tolist()
        
    # 4. Pass the raw dictionary through msgspec for instant validation
    # This automatically triggers your tag routing and array-length __post_init__ logic
    return msgspec.convert(bopp_data, type=Annotation, dec_hook=decode_arrow)


def load_bopp_csv(filepath: str | Path) -> Annotation:
    """
    End-to-end wrapper: Reads a BOPP CSV file directly into a validated 
    Annotation struct.
    """
    df = read_bopp_csv(filepath)
    return from_dataframe(df)
