"""Regression tests for signed and nonnegative fractional score values."""

import json
from pathlib import Path

import msgspec
import pytest

import bopp
from bopp.io import (
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.models.v1.extent.score_interval import ScoreInterval
from bopp.models.v1.extent.score_quarter import ScoreQuarterNotes


@pytest.fixture(scope="module")
def core_schema():
    """Load the authoritative core schema."""
    return json.loads(
        (Path(__file__).resolve().parents[1] / "schemas/v1/core.json").read_text()
    )


@pytest.mark.parametrize(
    ("definition", "value", "expected_valid"),
    [
        ("Fraction", [0, 1], True),
        ("Fraction", [-3, 4], True),
        ("Fraction", [1, 0], False),
        ("Fraction", [1, 2, 3], False),
        ("Fraction", [1], False),
        ("FractionNonnegative", [0, 1], True),
        ("FractionNonnegative", [-1, 2], False),
    ],
    ids=lambda value: repr(value),
)
def test_fraction_schema_values(definition, value, expected_valid, core_schema):
    """Fraction schemas enforce each element's bounds and the array length."""
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(
        {"$defs": core_schema["$defs"], "$ref": f"#/$defs/{definition}"}
    )
    assert validator.is_valid(value) == expected_valid


@pytest.mark.parametrize("quarter", [[[0, 1]], [[-1, 2]]])
def test_score_quarter_accepts_signed_numerator(quarter):
    """Score quarters accept zero and negative numerators as tuples."""
    decoded = msgspec.json.decode(
        msgspec.json.encode({"extent_type": "score_quarter", "quarter": quarter}),
        type=ScoreQuarterNotes,
    )
    assert decoded.quarter == [tuple(value) for value in quarter]


@pytest.mark.parametrize("quarter", [[[1, 0]], [[1, -2]], [[1, 2, 3]], [[1]]])
def test_score_quarter_rejects_invalid_fraction(quarter):
    """Score quarters reject nonpositive denominators and invalid lengths."""
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(
            msgspec.json.encode({"extent_type": "score_quarter", "quarter": quarter}),
            type=ScoreQuarterNotes,
        )


def test_score_interval_accepts_zero_duration():
    """Score intervals accept zero duration and a zero start quarter."""
    decoded = msgspec.json.decode(
        b'{"extent_type": "score_interval", "quarter": [[0, 1]], "duration": [[0, 1]]}',
        type=ScoreInterval,
    )
    assert decoded.quarter == [(0, 1)]
    assert decoded.duration == [(0, 1)]


def test_score_interval_rejects_negative_duration():
    """Score intervals reject negative duration numerators."""
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(
            b'{"extent_type": "score_interval", "quarter": [[0, 1]], "duration": [[-1, 2]]}',
            type=ScoreInterval,
        )


def test_create_and_validate_fraction_annotation():
    """Creating an annotation with zero and negative quarters passes validation."""
    annotation = bopp.create(
        media_id="test:frac",
        payload_kind="tag_open",
        extent_kind="score_quarter",
        quarter=[[0, 1], [-1, 2], [3, 4]],
        value=["a", "b", "c"],
    )
    assert bopp.validate(annotation)


@pytest.mark.parametrize(
    ("save", "load", "suffix", "keeps_id"),
    [
        (save_bopp_json, load_bopp_json, "json", True),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack", True),
        (save_bopp_csv, load_bopp_csv, "csv", False),
    ],
)
def test_fraction_annotation_roundtrip(tmp_path, save, load, suffix, keeps_id):
    """All formats preserve fractions; JSON and msgpack also preserve the ID."""
    quarter = [[0, 1], [-1, 2], [3, 4]]
    annotation = bopp.create(
        media_id="test:frac",
        payload_kind="tag_open",
        extent_kind="score_quarter",
        quarter=quarter,
        value=["a", "b", "c"],
    )
    path = tmp_path / f"fraction.{suffix}"
    save(annotation, path)
    loaded = load(path)
    assert [list(value) for value in loaded.extent.quarter] == quarter
    if keeps_id:
        assert loaded.id == annotation.id


def test_json_loaded_fraction_annotation_csv_roundtrip(tmp_path):
    """CSV preserves tuple fractions decoded from a saved JSON annotation."""
    quarter = [[0, 1], [-1, 2], [3, 4]]
    annotation = bopp.create(
        media_id="test:frac",
        payload_kind="tag_open",
        extent_kind="score_quarter",
        quarter=quarter,
        value=["a", "b", "c"],
    )
    json_path = tmp_path / "fraction.json"
    save_bopp_json(annotation, json_path)
    decoded = load_bopp_json(json_path)
    assert all(isinstance(value, tuple) for value in decoded.extent.quarter)
    csv_path = tmp_path / "fraction.csv"
    save_bopp_csv(decoded, csv_path)
    loaded = load_bopp_csv(csv_path)
    assert [list(value) for value in loaded.extent.quarter] == quarter
