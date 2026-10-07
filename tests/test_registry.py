"""Regression tests for complex column detection and fraction CSV loading."""

import ast
import importlib.util
import typing
from pathlib import Path
from typing import Annotated

import pytest
from msgspec import Meta

import bopp
from bopp.io import load_bopp_csv, save_bopp_csv
from bopp.models.v1.extent.score_interval import ScoreInterval
from bopp.registries.v1 import COMPLEX_FIELDS_REGISTRY

_spec = importlib.util.spec_from_file_location(
    "generate_registry",
    Path(__file__).resolve().parents[1] / "scripts" / "generate_registry.py",
)
generate_registry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_registry)

A = typing.TypeAliasType("A", Annotated[tuple[int, int], Meta(min_length=2)])
B = typing.TypeAliasType("B", Annotated[list[int], Meta(min_length=2)])
S = typing.TypeAliasType("S", Annotated[str, Meta(pattern="^a")])


def test_score_interval_hints():
    """Both score interval fraction columns require complex value parsing."""
    hints = typing.get_type_hints(ScoreInterval, include_extras=True)

    assert generate_registry._is_complex_field(hints["quarter"])
    assert generate_registry._is_complex_field(hints["duration"])


def test_alias_element_types():
    """Container aliases are complex while scalar aliases remain scalar."""
    assert generate_registry._is_complex_field(list[A])
    assert generate_registry._is_complex_field(list[B])
    assert not generate_registry._is_complex_field(list[float])
    assert not generate_registry._is_complex_field(list[str])
    assert not generate_registry._is_complex_field(list[S])


def test_nested_aliases():
    """Repeated alias and Annotated wrappers preserve column element detection."""
    container = typing.TypeAliasType("Container", Annotated[A, Meta()])
    scalar = typing.TypeAliasType("Scalar", Annotated[S, Meta()])
    column = typing.TypeAliasType("Column", Annotated[list[container], Meta()])
    scalar_column = typing.TypeAliasType("ScalarColumn", Annotated[list[scalar], Meta()])

    assert generate_registry._is_complex_field(column)
    assert not generate_registry._is_complex_field(scalar_column)


@pytest.mark.parametrize("name", ["Fraction", "FractionNonnegative"])
@pytest.mark.parametrize("prefix", ["", "core."])
def test_fraction_ast_fallback(name, prefix):
    """The AST fallback recognises bare and qualified fraction variant names."""
    node = ast.parse(f"list[{prefix}{name}]", mode="eval").body

    assert generate_registry._is_complex_ast_node(node)


@pytest.mark.parametrize(
    "source",
    [
        "list[core.FractionItem]",
        "list[FractionNonnegativeItem1]",
        "list[float]",
        'Annotated[list[str], Meta(description="x")]',
    ],
)
def test_scalar_ast_fallback(source):
    """The AST fallback leaves scalar element types unmarked."""
    node = ast.parse(source, mode="eval").body

    assert not generate_registry._is_complex_ast_node(node)


def test_score_interval_registry():
    """The generated registry includes both score interval fraction columns."""
    fields = COMPLEX_FIELDS_REGISTRY["extent_type"]["score_interval"]

    assert "quarter" in fields
    assert "duration" in fields


def test_score_interval_csv_roundtrip(tmp_path):
    """CSV loading restores both score interval fraction columns."""
    ann = bopp.create(
        media_id="test:reg",
        payload_kind="tag_open",
        extent_kind="score_interval",
        quarter=[[1, 2], [3, 4]],
        duration=[[1, 4], [7, 8]],
        value=["a", "b"],
    )
    path = tmp_path / "score_interval.csv"

    save_bopp_csv(ann, path)
    loaded = load_bopp_csv(path)

    assert [list(x) for x in loaded.extent.quarter] == [list(x) for x in ann.extent.quarter]
    assert [list(x) for x in loaded.extent.duration] == [list(x) for x in ann.extent.duration]
