#!/usr/bin/env python
import argparse
import ast
from collections import defaultdict
from pathlib import Path

def generate_registry(input_dir: Path, output_file: Path, base_module: str) -> None:
    # Registries structure: { tag_field: { tag_value: class_name } }
    registries = defaultdict(dict)
    # Imports structure: { submodule_import_path: set(class_names) }
    imports_by_module = defaultdict(set)

    # Walk all .py files recursively in the generated schema tree
    for py_file in sorted(input_dir.rglob("*.py")):
        # Skip top-level package init files or the target registry file itself
        if py_file.resolve() == output_file.resolve() or py_file.name == "__init__.py":
            continue

        # Determine the relative module import path (e.g. bopp.models.v1.payload.chord)
        rel_path = py_file.relative_to(input_dir).with_suffix("")
        submodule = f"{base_module}.{'.'.join(rel_path.parts)}"

        tree = ast.parse(py_file.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                tag_field = None
                tag_value = None

                for kw in node.keywords:
                    if kw.arg == "tag_field" and isinstance(kw.value, ast.Constant):
                        tag_field = kw.value.value
                    elif kw.arg == "tag" and isinstance(kw.value, ast.Constant):
                        tag_value = kw.value.value

                if tag_field and tag_value:
                    registries[tag_field][tag_value] = node.name
                    imports_by_module[submodule].add(node.name)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("# AUTO-GENERATED: Do not edit manually.\n\n")

        # Emit explicit submodule imports
        for mod_path in sorted(imports_by_module.keys()):
            classes = ", ".join(sorted(imports_by_module[mod_path]))
            f.write(f"from {mod_path} import {classes}\n")
        f.write("\n")

        # Emit dictionary registries per tag_field
        for tag_field in sorted(registries.keys()):
            tags = registries[tag_field]
            dict_name = f"{tag_field.upper()}_REGISTRY"
            f.write(f"{dict_name} = {{\n")
            for tag in sorted(tags.keys()):
                f.write(f"    {repr(tag)}: {tags[tag]},\n")
            f.write("}\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to models root folder")
    parser.add_argument("--output", type=Path, required=True, help="Destination registry file path")
    parser.add_argument("--base-module", type=str, required=True, help="Base Python import prefix")
    args = parser.parse_args()

    generate_registry(args.input_dir, args.output, args.base_module)

if __name__ == "__main__":
    main()
