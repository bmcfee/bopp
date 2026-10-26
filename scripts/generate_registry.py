#!/usr/bin/env python
import argparse
import ast
import importlib
import sys
import types
import typing
from collections import defaultdict
from pathlib import Path
from typing import Any, get_args, get_origin

_FRACTION_TYPES = ("Fraction", "FractionNonnegative")


def _unwrap_type(tp: Any) -> Any:
    """Recursively peel TypeAliasType, Annotated, and single-type Optional unions."""
    while True:
        if isinstance(tp, typing.TypeAliasType):
            tp = tp.__value__
        elif hasattr(tp, "__metadata__"):
            args = get_args(tp)
            tp = args[0] if args else tp
        elif get_origin(tp) in (typing.Union, types.UnionType):
            non_empty_args = [
                arg
                for arg in get_args(tp)
                if arg not in (type(None), getattr(importlib.import_module("msgspec"), "UnsetType", type(None)))
            ]
            if len(non_empty_args) == 1:
                tp = non_empty_args[0]
            else:
                break
        else:
            break
    return tp


def _is_complex_type(tp: Any) -> bool:
    """Determine if a type hint represents a complex structure (list, dict, tuple, Fraction, Any, etc.)."""
    tp = _unwrap_type(tp)
    if tp is Any:
        return True

    origin = get_origin(tp)

    if origin in (list, dict, tuple, set, getattr(__builtins__, "frozenset", set)):
        return True

    if origin in (typing.Union, types.UnionType):
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        return any(_is_complex_type(arg) for arg in args)

    if isinstance(tp, type):
        return not issubclass(tp, (int, float, str, bool, bytes))

    tp_str = str(tp)
    return any(k in tp_str for k in ("Fraction", "list", "dict", "tuple", "Any"))


def _is_string_type(tp: Any) -> bool:
    """Determine if a type hint represents a string type (including Annotated and TypeAliasType)."""
    tp = _unwrap_type(tp)

    origin = get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        return any(_is_string_type(arg) for arg in args)

    if isinstance(tp, type):
        return issubclass(tp, str)

    return False


def _is_complex_field(type_hint: Any) -> bool:
    """Check if the inner element type of a list array field is complex."""
    type_hint = _unwrap_type(type_hint)
    origin = get_origin(type_hint)

    if origin in (list, getattr(importlib.import_module("typing"), "Sequence", list)):
        args = get_args(type_hint)
        if args:
            inner_type = args[0]
            return _is_complex_type(inner_type)

    return _is_complex_type(type_hint)


def _is_string_field(type_hint: Any) -> bool:
    """Check if the inner element type of a list array field is a string."""
    type_hint = _unwrap_type(type_hint)
    origin = get_origin(type_hint)

    if origin in (list, getattr(importlib.import_module("typing"), "Sequence", list)):
        args = get_args(type_hint)
        if args:
            inner_type = args[0]
            return _is_string_type(inner_type)

    return _is_string_type(type_hint)


def _is_complex_ast_node(node: ast.AST) -> bool:
    """Fallback AST check for complex field types (Fraction, list, dict, tuple, object, Any)."""
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Subscript):
            return True
        if isinstance(node.slice, ast.Attribute) and node.slice.attr in _FRACTION_TYPES:
            return True
        if isinstance(node.slice, ast.Name) and (
            node.slice.id in _FRACTION_TYPES or node.slice.id in ("dict", "list", "tuple", "Any", "object")
        ):
            return True

    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in _FRACTION_TYPES:
            return True
        if isinstance(child, ast.Name) and (
            child.id in _FRACTION_TYPES or child.id in ("dict", "tuple", "Any", "object")
        ):
            return True

    return False


def _resolve_absolute_module(output_file: Path, base_module: str, sub_parts: list[str]) -> str:
    """Convert relative base_module (e.g. ..models.v1) into absolute bopp.models.v1 path."""
    if not base_module.startswith("."):
        return f"{base_module}.{'.'.join(sub_parts)}" if sub_parts else base_module

    clean_base = base_module.lstrip(".")
    dots_count = len(base_module) - len(clean_base)

    output_abs = output_file.resolve().parent
    for _ in range(dots_count - 1):
        output_abs = output_abs.parent

    rel_from_cwd = output_abs.relative_to(Path.cwd())
    pkg_parts = list(rel_from_cwd.parts)

    if clean_base:
        pkg_parts.append(clean_base)
    pkg_parts.extend(sub_parts)

    return ".".join(pkg_parts)


def generate_registry(input_dir: Path, output_file: Path, base_module: str) -> None:
    registries = defaultdict(dict)
    complex_fields = defaultdict(dict)
    string_fields = defaultdict(dict)
    imports_by_module = defaultdict(set)
    annotation_class_info = None
    grouping_class_info = None

    input_dir = input_dir.resolve()

    for py_file in sorted(input_dir.rglob("*.py")):
        if py_file.resolve() == output_file.resolve() or py_file.name == "__init__.py":
            continue

        rel_path = py_file.relative_to(input_dir).with_suffix("")
        sub_parts = [p for p in rel_path.parts if p != "."]

        rel_submodule = f"{base_module}.{'.'.join(sub_parts)}" if sub_parts else base_module
        abs_submodule = _resolve_absolute_module(output_file, base_module, sub_parts)

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
                    imports_by_module[rel_submodule].add(node.name)

                    complex_cols = []
                    string_cols = []

                    try:
                        mod = importlib.import_module(abs_submodule)
                        cls = getattr(mod, node.name)
                        type_hints = typing.get_type_hints(cls, globalns=mod.__dict__, include_extras=True)

                        for field_name, hint in type_hints.items():
                            if field_name == tag_field:
                                continue
                            if _is_complex_field(hint):
                                complex_cols.append(field_name)
                            elif _is_string_field(hint):
                                string_cols.append(field_name)
                    except Exception as err:  # noqa: BLE001
                        sys.stderr.write(
                            f"Warning: Failed to inspect annotations for {node.name} in {abs_submodule}: {err}\n"
                        )

                    if not complex_cols:
                        for stmt in node.body:
                            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                                field_name = stmt.target.id
                                if field_name == tag_field:
                                    continue
                                if _is_complex_ast_node(stmt.annotation):
                                    complex_cols.append(field_name)

                    if complex_cols:
                        complex_fields[tag_field][tag_value] = complex_cols
                    if string_cols:
                        string_fields[tag_field][tag_value] = string_cols

                elif node.name == "Annotation":
                    annotation_class_info = node.name
                    imports_by_module[rel_submodule].add(node.name)
                elif node.name == "Grouping":
                    grouping_class_info = node.name
                    imports_by_module[rel_submodule].add(node.name)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("# AUTO-GENERATED: Do not edit manually.\n\n")

        for mod_path in sorted(imports_by_module.keys()):
            classes = ", ".join(sorted(imports_by_module[mod_path]))
            f.write(f"from {mod_path} import {classes}\n")
        f.write("\n")

        all_exports = []

        if annotation_class_info:
            f.write(f"ANNOTATION_CLASS = {annotation_class_info}\n\n")
            all_exports.append("ANNOTATION_CLASS")

        if grouping_class_info:
            f.write(f"GROUPING_CLASS = {grouping_class_info}\n\n")
            all_exports.append("GROUPING_CLASS")

        for tag_field in sorted(registries.keys()):
            tags = registries[tag_field]
            dict_name = f"{tag_field.upper()}_REGISTRY"
            all_exports.append(dict_name)
            f.write(f"{dict_name} = {{\n")
            for tag in sorted(tags.keys()):
                f.write(f"    {tag!r}: {tags[tag]},\n")
            f.write("}\n\n")

        all_exports.append("COMPLEX_FIELDS_REGISTRY")
        f.write("COMPLEX_FIELDS_REGISTRY: dict[str, dict[str, list[str]]] = {\n")
        for tag_field in sorted(complex_fields.keys()):
            f.write(f"    {tag_field!r}: {{\n")
            for tag, fields in sorted(complex_fields[tag_field].items()):
                f.write(f"        {tag!r}: {fields!r},\n")
            f.write("    },\n")
        f.write("}\n\n")

        all_exports.append("STRING_FIELDS_REGISTRY")
        f.write("STRING_FIELDS_REGISTRY: dict[str, dict[str, list[str]]] = {\n")
        for tag_field in sorted(string_fields.keys()):
            f.write(f"    {tag_field!r}: {{\n")
            for tag, fields in sorted(string_fields[tag_field].items()):
                f.write(f"        {tag!r}: {fields!r},\n")
            f.write("    },\n")
        f.write("}\n\n")

        all_exports.sort()
        f.write(f"__all__ = {all_exports!r}\n")

    import subprocess

    subprocess.run(["ruff", "format", str(output_file)], check=False, capture_output=True)


def main():
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to models root folder")
    parser.add_argument("--output", type=Path, required=True, help="Destination registry file path")
    parser.add_argument("--base-module", type=str, required=True, help="Base Python import prefix")
    args = parser.parse_args()

    generate_registry(args.input_dir, args.output, args.base_module)


if __name__ == "__main__":
    main()
