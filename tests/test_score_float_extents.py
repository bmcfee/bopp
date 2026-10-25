"""Validate float score extents, serialization, and numeric trimming."""

import copy
from fractions import Fraction

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
from bopp.models.v1.extent.score_interval_float import ScoreIntervalFloat
from bopp.models.v1.extent.score_quarter_float import ScoreQuarterNotesFloat
from bopp.registries.v1 import COMPLEX_FIELDS_REGISTRY, EXTENT_TYPE_REGISTRY
from bopp.transforms import AXIS_CONFIGS, DEFAULT_TARGET_FIELDS, FRACTION_AXES

EXTENTS = [
    ("score_quarter_float", ScoreQuarterNotesFloat, {"quarter": [-1.5, 0.0, 2.25]}),
    (
        "score_interval_float",
        ScoreIntervalFloat,
        {"quarter": [-1.5, 0.0, 2.25], "duration": [0.0, 0.5, 1.25]},
    ),
]


@pytest.mark.parametrize("tag, cls, columns", EXTENTS)
def test_decode_valid_score_float(tag, cls, columns):
    """Signed positions and nonnegative durations decode and validate."""
    decoded = msgspec.json.decode(
        msgspec.json.encode({"extent_type": tag, **columns}), type=cls
    )
    for name, expected in columns.items():
        assert getattr(decoded, name) == expected
    ann = bopp.create(
        media_id="test:score", payload_kind="tag_open", extent_kind=tag,
        value=["pickup", "start", "later"], **columns,
    )
    assert bopp.validate(ann)
    assert EXTENT_TYPE_REGISTRY[tag] is cls
    assert tag not in COMPLEX_FIELDS_REGISTRY["extent_type"]


@pytest.mark.parametrize("tag, cls, columns", EXTENTS)
def test_score_float_rejects_fraction_columns(tag, cls, columns):
    """Each float column rejects fraction pairs during decoding."""
    for name in columns:
        invalid = copy.deepcopy(columns)
        invalid[name][0] = [1, 2]
        with pytest.raises(msgspec.ValidationError, match=rf"\$\.{name}"):
            msgspec.json.decode(
                msgspec.json.encode({"extent_type": tag, **invalid}), type=cls,
            )


def test_score_float_rejects_negative_duration():
    with pytest.raises(msgspec.ValidationError, match=r"\$\.duration"):
        msgspec.json.decode(
            b'{"extent_type":"score_interval_float","quarter":[-1.0],"duration":[-0.5]}',
            type=ScoreIntervalFloat,
        )
    with pytest.raises(bopp.BoppValidationError):
        bopp.validate(
            {"extent_type": "score_interval_float", "quarter": [-1.0], "duration": [-0.5]},
            target_type=ScoreIntervalFloat,
        )


@pytest.mark.parametrize("tag, cls, columns", EXTENTS)
@pytest.mark.parametrize("save, load, suffix, keeps_id", [
    (save_bopp_json, load_bopp_json, "json", True),
    (save_bopp_msgpack, load_bopp_msgpack, "msgpack", True),
    (save_bopp_csv, load_bopp_csv, "csv", False),
])
def test_score_float_roundtrip(tmp_path, tag, cls, columns, save, load, suffix, keeps_id):
    ann = bopp.create(
        media_id="test:score", payload_kind="tag_open", extent_kind=tag,
        value=["pickup", "start", "later"], **columns,
    )
    path = tmp_path / f"{tag}.{suffix}"
    save(ann, path)
    loaded = load(path)
    assert bopp.validate(loaded)
    assert isinstance(loaded.extent, cls)
    assert loaded.media_id == ann.media_id
    assert loaded.payload == ann.payload
    for name, expected in columns.items():
        assert getattr(loaded.extent, name) == expected
    if keeps_id:
        assert loaded.id == ann.id


@pytest.mark.parametrize("interval", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_trim_score_float_matches_time(interval, strict, reset):
    tag = "score_interval_float" if interval else "score_quarter_float"
    positions = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    kwargs = {"duration": [1.0, 1.0, 0.5, 1.0, 0.0, 1.0]} if interval else {}
    common = dict(media_id="test:score", payload_kind="tag_open", value=list("abcdef"), **kwargs)
    ann = bopp.create(extent_kind=tag, quarter=positions, **common)
    timed = bopp.create(extent_kind="time_interval" if interval else "time", time=positions, **common)
    original = copy.deepcopy(ann)
    bounds = {"start": 0.5, "end": 1.5, "strict": strict, "reset": reset}
    actual = bopp.trim(ann, **bounds)
    expected = bopp.trim(timed, **bounds)
    assert actual.extent.quarter == expected.extent.time
    if interval:
        assert actual.extent.duration == expected.extent.duration
    assert actual.payload == expected.payload
    assert bopp.validate(actual)
    assert ann == original
    assert DEFAULT_TARGET_FIELDS[tag] == "quarter"
    assert AXIS_CONFIGS[(tag, "quarter")] == (
        "origin_span" if interval else "point", "quarter", "duration" if interval else None,
    )
    assert (tag, "quarter") not in FRACTION_AXES


@pytest.mark.parametrize("tag, cls, columns", EXTENTS)
@pytest.mark.parametrize("bound", [[1, 2], (1, 2), Fraction(1, 2)])
@pytest.mark.parametrize("name", ["start", "end"])
def test_trim_score_float_rejects_fraction_bounds(tag, cls, columns, bound, name):
    ann = bopp.create(
        media_id="test:score", payload_kind="tag_open", extent_kind=tag,
        value=list("abc"), **columns,
    )
    with pytest.raises(bopp.BoppArgumentError, match="bounds require a fraction axis"):
        bopp.trim(ann, **{name: bound})
