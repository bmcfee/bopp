import json
from pathlib import Path

from scripts.build_schema_site import (
    build_site,
    get_version_dirs,
    resolve_dev_dir,
    validate_schema_references,
)


def test_get_version_dirs():
    schemas_dir = Path("schemas")
    versions = get_version_dirs(schemas_dir)
    assert "v1" in versions
    assert "dev" not in versions


def test_resolve_dev_dir():
    schemas_dir = Path("schemas")
    versions = get_version_dirs(schemas_dir)
    dev_target = resolve_dev_dir(schemas_dir, versions)
    assert dev_target.name == "v1"


def test_build_schema_site(tmp_path: Path):
    output_dir = tmp_path / "_site"
    build_site(schemas_dir=Path("schemas"), output_dir=output_dir)

    # 1. Verify marker and index files
    assert (output_dir / ".nojekyll").is_file()
    assert (output_dir / "index.html").is_file()
    assert (output_dir / "dev" / "index.html").is_file()
    assert (output_dir / "v1" / "index.html").is_file()

    root_html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert 'href="dev/"' in root_html
    assert 'href="v1/"' in root_html

    # 2. Verify schema files in dev and v1
    for version in ("dev", "v1"):
        ann_file = output_dir / version / "annotation.json"
        assert ann_file.is_file()
        ann_data = json.loads(ann_file.read_text(encoding="utf-8"))
        assert ann_data["$id"] == "annotation.json"

        core_file = output_dir / version / "core.json"
        assert core_file.is_file()
        core_data = json.loads(core_file.read_text(encoding="utf-8"))
        assert core_data["$id"] == "core.json"

        extent_file = output_dir / version / "extent" / "times.json"
        assert extent_file.is_file()

    # 3. Verify no broken references
    errors = validate_schema_references(output_dir)
    assert errors == []


def test_validate_schema_references_broken(tmp_path: Path):
    bad_dir = tmp_path / "broken"
    bad_dir.mkdir()
    bad_schema = bad_dir / "test.json"
    bad_schema.write_text(json.dumps({"$ref": "non_existent.json"}), encoding="utf-8")

    errors = validate_schema_references(bad_dir)
    assert len(errors) == 1
    assert "non_existent.json not found" in errors[0]
