import ast
from pathlib import Path

import msgspec
import pandas as pd
import tomllib

from bopp.models.v1.annotation import Annotation

from .util import extract_header, to_dataframe


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
        df.to_csv(f, index=False)


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


def read_bopp_csv(filepath: str | Path) -> pd.DataFrame:
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


def from_dataframe(df) -> Annotation:
    """
    Reconstitute a strictly typed Annotation struct from a DataFrame.

    Expects singleton fields (like media_id, metadata) to be in `df.attrs`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing BOPP tabular columns and frontmatter attributes.

    Returns
    -------
    Annotation
        Validated Annotation struct constructed from the DataFrame.
    """
    # 1. Scaffold the base dictionary with required singletons
    bopp_data = {
        "bopp_version": df.attrs.get("bopp_version", "1.0.0"),
        "media_id": df.attrs.get("media_id", "unknown:media"),
        "payload": {}
    }
    
    # Safely inject optional root fields if they exist in the TOML header
    if "metadata" in df.attrs:
        bopp_data["metadata"] = df.attrs["metadata"]
    if "annotated_domain" in df.attrs:
        bopp_data["annotated_domain"] = df.attrs["annotated_domain"]
        
    # 2. Infer structural types from the self-describing column headers (facet:type:field_name)
    coord_cols = [c for c in df.columns if c.startswith("extent:")]
    payload_cols = [c for c in df.columns if c.startswith("payload:")]
    conf_cols = [c for c in df.columns if c.startswith("confidence:")]
    
    if coord_cols:
        ext_type = coord_cols[0].split(":")[1]
        bopp_data["extent"] = {"extent_type": ext_type}
        for col in coord_cols:
            parts = col.split(":")
            field_name = parts[2] if len(parts) > 2 else "values"
            bopp_data["extent"][field_name] = df[col].tolist()

    if payload_cols:
        payload_type = payload_cols[0].split(":")[1]
        bopp_data["payload"]["payload_type"] = payload_type
        for col in payload_cols:
            parts = col.split(":")
            field_name = parts[2] if len(parts) > 2 else "values"
            bopp_data["payload"][field_name] = df[col].tolist()

    if conf_cols:
        conf_type = conf_cols[0].split(":")[1]
        bopp_data["confidence"] = {"confidence_type": conf_type}
        for col in conf_cols:
            parts = col.split(":")
            field_name = parts[2] if len(parts) > 2 else "confidence"
            bopp_data["confidence"][field_name] = df[col].tolist()
        
    # 3. Pass the raw dictionary through msgspec for instant validation
    # This automatically triggers your tag routing and array-length __post_init__ logic
    return msgspec.convert(bopp_data, type=Annotation)


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
