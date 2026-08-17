#!/usr/bin/env python

import pandas as pd
import numpy as np
import msgspec

from typing import Any

# Import your generated msgspec models here
from .models.v1.annotation import Annotation 

# ==========================================
# 1. Struct <-> DataFrame Translators
# ==========================================

def _get_tag(struct: msgspec.Struct) -> str | None:
    """
    Retrieves the tag value from a msgspec Struct, if it has one.
    """
    config = getattr(struct, "__struct_config__", None)
    if config is not None:
        return getattr(config, "tag", None)
    return None


def extract_header(annotation: Annotation) -> dict[str, Any]:
     """
     Extracts all singleton fields from a Annotation for YAML serialization,
     explicitly omitting the parallel array blocks.
     """
     # The fields that belong in the CSV body, not the YAML header
     excluded_fields = {"extent", "payload", "confidence"}
 
     header_data = {}
 
     # Iterate over all fields defined on the msgspec Struct
     for field in msgspec.structs.fields(annotation):
         if field.name in excluded_fields:
             continue
 
         value = getattr(annotation, field.name)
 
         # Skip optional fields that weren't populated (e.g., metadata or annotated_domain)
         if value is not None and value is not msgspec.UNSET:
             # Safely convert nested msgspec structs into standard Python dictionaries 
             # and lists, ensuring they are 100% compatible with yaml.dump()
             header_data[field.name] = msgspec.to_builtins(value)
 
     return header_data


def to_dataframe(annotation: Annotation) -> pd.DataFrame:
    """
    Converts a Annotation into a Pandas DataFrame using self-describing 
    column headers. Singleton metadata is preserved in df.attrs.
    """
    data = {}
    
    # 1. Parse Extents (Geometry)
    ext_type = _get_tag(annotation.extent)
    coords = np.array(annotation.extent.coordinates)
    num_coord_dims = coords.shape[1] if coords.ndim > 1 else 1
    
    for i in range(num_coord_dims):
        col_name = f"extent:{ext_type}:{i}"
        data[col_name] = coords[:, i] if num_coord_dims > 1 else coords
        
    # 2. Parse Payload (Passenger Data)
    payload_type = _get_tag(annotation.payload)
    data[f"payload:{payload_type}"] = annotation.payload.values
    
    # 3. Parse Confidence (if present)
    if annotation.confidence is not None and annotation.confidence is not msgspec.UNSET:
        conf_type = _get_tag(annotation.confidence)
        data[f"confidence:{conf_type}"] = annotation.confidence.confidence
        
    df = pd.DataFrame(data)
    
    # 4. Stash the singleton data safely into the DataFrame's attributes
    df.attrs["bopp_version"] = annotation.bopp_version
    df.attrs["media_id"] = annotation.media_id
    if annotation.metadata:
        # to_builtins safely converts nested msgspec Structs into standard dicts
        df.attrs["metadata"] = msgspec.to_builtins(annotation.metadata)
        
    return df


def from_dataframe(df: pd.DataFrame) -> Annotation:
    """
    Reconstitutes a strictly typed Annotation struct from a DataFrame.
    Expects singletons to be present in df.attrs.
    """
    # 1. Scaffold the base dictionary from df.attrs
    bopp_data = {
        "bopp_version": df.attrs.get("bopp_version", "1.0.0"),
        "media_id": df.attrs.get("media_id", "unknown:media"),
        "extent": {"coordinates": []},
        "payload": {"values": []}
    }
    
    if "metadata" in df.attrs:
        bopp_data["metadata"] = df.attrs["metadata"]
        
    # 2. Infer types from the self-describing column headers
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
        bopp_data["extent"]["coordinates"] = df[coord_cols[0]].tolist()
    else:
        # Stack 2D+ coordinates (like time_intervals) back into lists of lists
        bopp_data["extent"]["coordinates"] = df[coord_cols].to_numpy().tolist()
        
    # 4. Pass the raw dictionary through msgspec for instant validation
    return msgspec.convert(bopp_data, type=Annotation)
