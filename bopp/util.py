#!/usr/bin/env python

import pandas as pd
import numpy as np
import msgspec

from typing import Any

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
     Extracts all singleton fields from an Annotation for YAML serialization,
     explicitly omitting the parallel array blocks.
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


def to_dataframe(annotation: Annotation) -> pd.DataFrame:
    """
    Converts an Annotation into a Pandas DataFrame using self-describing 
    column headers. Singleton metadata is preserved in df.attrs.
    """
    data = {}
    
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
        
    df = pd.DataFrame(data)
    
    # 4. Stash the singleton data safely into the DataFrame's attributes
    df.attrs["bopp_version"] = annotation.bopp_version
    df.attrs["media_id"] = annotation.media_id
    if annotation.metadata:
        df.attrs["metadata"] = msgspec.to_builtins(annotation.metadata)
        
    return df


def from_dataframe(df: pd.DataFrame) -> Annotation:
    """
    Reconstitutes a strictly typed Annotation struct from a DataFrame.
    Expects singletons to be present in df.attrs.
    """
    bopp_data = {
        "bopp_version": df.attrs.get("bopp_version", "1.0.0"),
        "media_id": df.attrs.get("media_id", "unknown:media"),
        "payload": {}
    }
    
    if "metadata" in df.attrs:
        bopp_data["metadata"] = df.attrs["metadata"]
        
    coord_cols = [c for c in df.columns if c.startswith("extent:")]
    payload_cols = [c for c in df.columns if c.startswith("payload:")]
    conf_cols = [c for c in df.columns if c.startswith("confidence:")]
    
    if coord_cols:
        ext_type = coord_cols[0].split(":")[1]
        bopp_data["extent"] = {"extent_type": ext_type}
        for col in coord_cols:
            field_name = col.split(":")[2]
            bopp_data["extent"][field_name] = df[col].tolist()

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
        
    return msgspec.convert(bopp_data, type=Annotation)
