"""Unit tests for codegen Jinja template synchronization."""

from pathlib import Path
import pytest


def test_codegen_templates_match_upstream():
    """Verify that templates in templates/ match the expected patched upstream templates."""
    pytest.importorskip("datamodel_code_generator")
    from scripts.sync_codegen_templates import get_target_templates

    template_dir = Path("templates")
    expected = get_target_templates()

    for name, content in expected.items():
        path = template_dir / name
        assert path.exists(), f"Template {name} is missing in templates/"
        assert path.read_text() == content, (
            f"Template {name} does not match expected upstream template. "
            "Run 'hatch run codegen:sync-templates' to update."
        )
