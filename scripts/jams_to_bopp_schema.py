#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def translate_jams_to_bopp(jams_path: str | Path, bopp_path: str | Path):
    """
    Translates a JAMS namespace JSON definition into a BOPP payload JSON schema.
    """
    # 1. Load the JAMS definition
    with open(jams_path, 'r') as f:
        jams_data = json.load(f)
        
    # 2. Extract the namespace key (e.g., 'chord') and its contents
    namespace_key = list(jams_data.keys())[0]
    jams_def = jams_data[namespace_key]
    
    # 3. Extract the value constraints and description
    jams_value_constraints = jams_def.get("value", {})
    description = jams_def.get("description", f"{namespace_key.title()} Payload")
    
    # 4. Construct the BOPP JSON Schema
    bopp_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"payload/{namespace_key}.json",
        "title": f"{namespace_key.title()} Payload",
        "type": "object",
        "properties": {
            "payload_type": {
                "const": namespace_key
            },
            "values": {
                "type": "array",
                "description": description,
                "items": jams_value_constraints
            }
        },
        "required": [
            "payload_type", 
            "values"
        ]
    }
    
    # 5. Write the generated schema to disk
    with open(bopp_path, 'w') as f:
        json.dump(bopp_schema, f, indent=2)
        
    print(f"Successfully translated '{namespace_key}' to {bopp_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JAMS namespaces to BOPP payload schemas.")
    parser.add_argument("input", help="Path to the input JAMS JSON file")
    parser.add_argument("output", help="Path to save the output BOPP JSON Schema file")
    
    args = parser.parse_args()
    translate_jams_to_bopp(args.input, args.output)
