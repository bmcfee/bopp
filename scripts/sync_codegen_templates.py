#!/usr/bin/env python3
"""Sync custom datamodel-codegen templates with upstream generator templates."""

from __future__ import annotations

import argparse
import pathlib
import sys

import datamodel_code_generator.model.base as dcg_base

BOPP_ARROW_TEMPLATE = """\
{%- macro apply_arrow(type_str, extras) -%}
{%- if extras and 'x_arrow_dtype' in extras -%}
{%- set dtype = extras['x_arrow_dtype'] -%}
{%- if type_str.startswith('Annotated[list[') -%}
Annotated[ArrowArray[Literal["{{ dtype }}"], {{ type_str[15:] }}
{%- elif type_str.startswith('list[') -%}
ArrowArray[Literal["{{ dtype }}"], {{ type_str[5:] }}
{%- else -%}
{{ type_str }}
{%- endif -%}
{%- else -%}
{{ type_str }}
{%- endif -%}
{%- endmacro -%}
"""


def generate_msgspec_template() -> str:
    upstream = (dcg_base.TEMPLATE_DIR / "msgspec.jinja2").read_text()
    patched = '{%- from "_bopp_arrow.jinja2" import apply_arrow -%}\n' + upstream
    patched = patched.replace("    {{ description }}", '    """{{ description }}"""')
    patched = patched.replace(
        "{{ field.annotated }}", "{{ apply_arrow(field.annotated, field.extras) }}"
    )
    patched = patched.replace(
        "{{ field.type_hint }}", "{{ apply_arrow(field.type_hint, field.extras) }}"
    )
    return patched


def generate_typestatement_template() -> str:
    upstream = (dcg_base.TEMPLATE_DIR / "TypeStatement.jinja2").read_text()
    patched = '{%- from "_bopp_arrow.jinja2" import apply_arrow -%}\n' + upstream
    patched = patched.replace("\n{{ description }}", '\n"""{{ description }}"""')
    patched = patched.replace(
        "{{ _field.annotated }}", "{{ apply_arrow(_field.annotated, _field.extras) }}"
    )
    patched = patched.replace(
        "{{ _field.type_hint }}", "{{ apply_arrow(_field.type_hint, _field.extras) }}"
    )
    return patched


def get_target_templates() -> dict[str, str]:
    return {
        "_bopp_arrow.jinja2": BOPP_ARROW_TEMPLATE,
        "msgspec.jinja2": generate_msgspec_template(),
        "TypeStatement.jinja2": generate_typestatement_template(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--write", action="store_true", help="Write synchronized templates to templates/"
    )
    group.add_argument(
        "--check", action="store_true", help="Check that templates/ are up to date"
    )
    parser.add_argument(
        "--template-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "templates",
        help="Directory containing custom templates (default: templates/)",
    )
    args = parser.parse_args()

    template_dir: pathlib.Path = args.template_dir
    expected = get_target_templates()

    if args.write:
        template_dir.mkdir(parents=True, exist_ok=True)
        for name, content in expected.items():
            path = template_dir / name
            path.write_text(content)
            print(f"Wrote {path}")
        return 0

    if args.check:
        mismatches: list[str] = []
        for name, content in expected.items():
            path = template_dir / name
            if not path.exists():
                mismatches.append(f"{name} is missing")
            elif path.read_text() != content:
                mismatches.append(f"{name} content does not match upstream template")
        if mismatches:
            print("Template synchronization check failed:")
            for m in mismatches:
                print(f"  - {m}")
            print("Run 'python scripts/sync_codegen_templates.py --write' to sync.")
            return 1
        print("Templates are synchronized with upstream datamodel-codegen.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
