#!/usr/bin/env python
import argparse
import ast
import importlib
import inspect
from collections import defaultdict
from pathlib import Path
from typing import Any, get_args, get_origin


def _is_complex_type(tp: Any) -> bool:
    """Determine if a type hint represents a complex structure (list, dict, tuple, Fraction, Any, etc.)."""
    if tp is Any:
        return True

    origin = get_origin(tp)

    # Handle Union / Optional
    if origin is getattr(type(int | str), "__origin__", None) or origin is Any:
        args = get_args(tp)
        return any(_is_complex_type(arg) for arg in args if arg is not type(None))

    if origin in (list, dict, tuple, set, getattr(__builtins__, "frozenset", set)):
        return True

    if origin is not None:
        args = get_args(tp)
        return any(_is_complex_type(arg) for arg in args if arg is not type(None))

    # Handles classes like dict, list, Struct subclasses, Fraction, etc.
    if isinstance(tp, type):
        if issubclass(tp, (int, float, str, bool, bytes)):
            return False
        return True

    # Check string representations for forward refs or type aliases like Fraction / dict
    tp_str = str(tp)
    if any(k in tp_str for k in ("Fraction", "list", "dict", "tuple", "Any")):
        return True

    return False


def _is_complex_field(type_hint: Any) -> bool:
    """Check if the inner element type of a list array field is complex."""
    origin = get_origin(type_hint)

    # Outer layer is typically list[...] for BOPP columnar arrays
    if origin in (list, getattr(importlib.import_module("typing"), "Sequence", list)):
        args = get_args(type_hint)
        if args:
            inner_type = args[0]
            return _is_complex_type(inner_type)

    return _is_complex_type(type_hint)


def generate_registry(input_dir: Path, output_file: Path, base_module: str) -> None:
    # Registries structure: { tag_field: { tag_value: class_name } }
    registries = defaultdict(dict)
    # Complex fields structure: { (tag_field, tag_value): [field_names] }
    complex_fields = defaultdict(dict)
    # Imports structure: { submodule_import_path: set(class_names) }
    imports_by_module = defaultdict(set)
    annotation_class_info = None

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

                    complex_cols = []
                    try:
                        mod = importlib.import_module(submodule)
                        cls = getattr(mod, node.name)
                        type_hints = inspect.get_annotations(cls, eval_str=True)

                        for field_name, hint in type_hints.items():
                            if field_name == tag_field:
                                continue
                            if _is_complex_field(hint):
                                complex_cols.append(field_name)
                    except Exception:
                        pass

                    if complex_cols:
                        complex_fields[tag_field][tag_value] = complex_cols

                elif node.name == "Annotation":
                    annotation_class_info = node.name
                    imports_by_module[submodule].add(node.name)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("# AUTO-GENERATED: Do not edit manually.\n\n")

        # Emit explicit submodule imports
        for mod_path in sorted(imports_by_module.keys()):
            classes = ", ".join(sorted(imports_by_module[mod_path]))
            f.write(f"from {mod_path} import {classes}\n")
        f.write("\n")

        all_exports = []

        if annotation_class_info:
            f.write(f"ANNOTATION_CLASS = {annotation_class_info}\n\n")
            all_exports.append("ANNOTATION_CLASS")

        # Emit dictionary registries per tag_field
        for tag_field in sorted(registries.keys()):
            tags = registries[tag_field]
            dict_name = f"{tag_field.upper()}_REGISTRY"
            all_exports.append(dict_name)
            f.write(f"{dict_name} = {{\n")
            for tag in sorted(tags.keys()):
                f.write(f"    {repr(tag)}: {tags[tag]},\n")
            f.write("}\n\n")

        # Emit COMPLEX_FIELDS_REGISTRY with type annotation
        all_exports.append("COMPLEX_FIELDS_REGISTRY")
        f.write("COMPLEX_FIELDS_REGISTRY: dict[str, dict[str, list[str]]] = {\n")
        for tag_field in sorted(complex_fields.keys()):
            f.write(f"    {repr(tag_field)}: {{\n")
            for tag, fields in sorted(complex_fields[tag_field].items()):
                f.write(f"        {repr(tag)}: {fields!r},\n")
            f.write("    },\n")
        f.write("}\n\n")

        all_exports.sort()
        f.write(f"__all__ = {all_exports!r}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to models root folder")
    parser.add_argument("--output", type=Path, required=True, help="Destination registry file path")
    parser.add_argument("--base-module", type=str, required=True, help="Base Python import prefix")
    args = parser.parse_args()

    generate_registry(args.input_dir, args.output, args.base_module)


if __name__ == "__main__":
    main()
