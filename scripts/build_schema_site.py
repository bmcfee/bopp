#!/usr/bin/env python3
"""Build the static site for hosting BOPP JSON schemas on GitHub Pages."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GitHub Pages static site for BOPP JSON schemas."
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=Path("schemas"),
        help="Path to the schemas directory in the repository (default: schemas)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_site"),
        help="Output directory for static site (default: _site)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip reference integrity validation on generated schemas",
    )
    return parser.parse_args()


def get_version_dirs(schemas_dir: Path) -> list[str]:
    """Find all version subdirectories in schemas (e.g. v1, v2)."""
    versions = []
    for entry in schemas_dir.iterdir():
        if entry.is_dir() and not entry.is_symlink() and entry.name.startswith("v"):
            versions.append(entry.name)
    return sorted(versions)


def resolve_dev_dir(schemas_dir: Path, versions: list[str]) -> Path:
    """Resolve the directory corresponding to the dev schema."""
    dev_path = schemas_dir / "dev"
    if dev_path.exists():
        return dev_path.resolve()
    if versions:
        return (schemas_dir / versions[-1]).resolve()
    raise FileNotFoundError(f"No schema versions found in {schemas_dir}")


def copy_schema_tree(src_dir: Path, dest_dir: Path) -> None:
    """Copy all json files from src_dir to dest_dir, preserving folder structure."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.glob("**/*.json"):
        rel_path = p.relative_to(src_dir)
        target = dest_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def collect_schema_info(version_dir: Path) -> list[dict]:
    """Collect metadata for all schema files in a version directory."""
    schemas = []
    for p in sorted(version_dir.glob("**/*.json")):
        rel_path = str(p.relative_to(version_dir))
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            title = data.get("title", rel_path)
            desc = data.get("description", "")
        except (json.JSONDecodeError, OSError):
            title = rel_path
            desc = ""
        schemas.append({"path": rel_path, "title": title, "description": desc})
    return schemas


def generate_version_index_html(version_name: str, schemas: list[dict]) -> str:
    """Generate index.html listing all schemas in a version."""
    rows = []
    for s in schemas:
        path = s["path"]
        title = s["title"]
        desc = s["description"]
        rows.append(
            f'<tr><td><a href="{path}"><code>{path}</code></a></td>'
            f"<td><strong>{title}</strong></td>"
            f"<td>{desc}</td></tr>"
        )
    rows_html = "\n      ".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BOPP JSON Schemas - {version_name}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0; padding: 2rem; color: #1f2328; background: #fff; line-height: 1.5;
    }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ border-bottom: 1px solid #d0d7de; padding-bottom: 0.5rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ text-align: left; padding: 0.5rem; border-bottom: 1px solid #d0d7de; }}
    th {{ background: #f6f8fa; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #f6f8fa; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.9em; }}
    .nav {{ margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="nav"><a href="../">&larr; BOPP Schemas Home</a></div>
    <h1>BOPP JSON Schemas ({version_name})</h1>
    <p>Index of available JSON schema files for version <code>{version_name}</code>.</p>
    <table>
      <thead>
        <tr>
          <th>Schema File</th>
          <th>Title</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
      {rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def generate_root_index_html(versions: list[str], dev_target_name: str) -> str:
    """Generate root index.html listing available versions."""
    version_items = [
        f'<li><a href="dev/"><strong>dev</strong></a> &mdash; Active development schema (currently mirrors <code>{dev_target_name}</code>)</li>'
    ]
    for v in reversed(versions):
        version_items.append(f'<li><a href="{v}/"><strong>{v}</strong></a></li>')
    version_list_html = "\n      ".join(version_items)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BOPP JSON Schemas</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0; padding: 2rem; color: #1f2328; background: #fff; line-height: 1.5;
    }}
    .container {{ max-width: 800px; margin: 0 auto; }}
    h1 {{ border-bottom: 1px solid #d0d7de; padding-bottom: 0.5rem; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #f6f8fa; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.9em; }}
    ul {{ line-height: 2; }}
    .card {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 1rem; margin-top: 1.5rem; }}
    pre {{ background: #eaeef2; padding: 0.75rem; border-radius: 4px; overflow-x: auto; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>BOPP JSON Schemas</h1>
    <p>This site hosts JSON schemas for the <strong>Bounded Observation Payload Protocol (BOPP)</strong>.</p>

    <h2>Available Versions</h2>
    <ul>
      {version_list_html}
    </ul>

    <div class="card">
      <h3>Quick Usage</h3>
      <p>Reference the primary annotation schema in JSON files or validators:</p>
      <pre><code>"$schema": "https://bmcfee.github.io/bopp/dev/annotation.json"</code></pre>
      <p>Or browse the full documentation at <a href="https://bopp.readthedocs.io/">bopp.readthedocs.io</a>.</p>
    </div>
  </div>
</body>
</html>
"""


def validate_schema_references(dest_dir: Path) -> list[str]:
    """Validate that all local $ref values point to existing files."""
    errors = []

    def check_refs(obj: object, file_path: Path, root_dir: Path) -> None:
        if isinstance(obj, dict):
            if "$ref" in obj and isinstance(obj["$ref"], str):
                ref = obj["$ref"]
                parsed = urlparse(ref)
                if not parsed.scheme and not parsed.netloc:
                    path_part = parsed.path
                    if path_part:
                        target = (file_path.parent / path_part).resolve()
                        if not target.exists():
                            rel_err = file_path.relative_to(dest_dir)
                            errors.append(
                                f"{rel_err}: broken $ref '{ref}' -> {path_part} not found"
                            )
            for v in obj.values():
                check_refs(v, file_path, root_dir)
        elif isinstance(obj, list):
            for item in obj:
                check_refs(item, file_path, root_dir)

    for p in dest_dir.glob("**/*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            check_refs(data, p, dest_dir)
        except (json.JSONDecodeError, OSError) as e:
            rel = p.relative_to(dest_dir)
            errors.append(f"{rel}: failed to load JSON: {e}")

    return errors


def build_site(
    schemas_dir: Path, output_dir: Path, skip_validation: bool = False
) -> None:
    """Build the complete static site."""
    schemas_dir = schemas_dir.resolve()
    output_dir = output_dir.resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. GitHub Pages .nojekyll marker
    (output_dir / ".nojekyll").touch()

    # 2. Find versions
    versions = get_version_dirs(schemas_dir)
    if not versions:
        raise RuntimeError(f"No version directories found in {schemas_dir}")

    # 3. Resolve dev
    dev_target = resolve_dev_dir(schemas_dir, versions)
    dev_target_name = dev_target.name

    # 4. Copy versions and generate version indexes
    for v in versions:
        v_dest = output_dir / v
        copy_schema_tree(schemas_dir / v, v_dest)
        schemas_info = collect_schema_info(v_dest)
        (v_dest / "index.html").write_text(
            generate_version_index_html(v, schemas_info), encoding="utf-8"
        )

    # 5. Copy dev and generate dev index
    dev_dest = output_dir / "dev"
    copy_schema_tree(dev_target, dev_dest)
    dev_schemas_info = collect_schema_info(dev_dest)
    (dev_dest / "index.html").write_text(
        generate_version_index_html(f"dev ({dev_target_name})", dev_schemas_info),
        encoding="utf-8",
    )

    # 6. Generate root index.html
    (output_dir / "index.html").write_text(
        generate_root_index_html(versions, dev_target_name), encoding="utf-8"
    )

    # 7. Validate references
    if not skip_validation:
        errors = validate_schema_references(output_dir)
        if errors:
            print("Validation errors encountered:")
            for err in errors:
                print(f"  - {err}")
            raise RuntimeError(
                f"Reference validation failed with {len(errors)} error(s)."
            )

    print(
        f"Successfully built schema site in {output_dir} (versions: {', '.join(versions)}, dev -> {dev_target_name})"
    )


def main() -> None:
    args = parse_args()
    try:
        build_site(
            schemas_dir=args.schemas_dir,
            output_dir=args.output_dir,
            skip_validation=args.skip_validation,
        )
    except (RuntimeError, OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
