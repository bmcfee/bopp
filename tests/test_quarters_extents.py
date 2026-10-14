"""Validate quarter-note extents and their annotation serialization."""

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
from bopp.models.v1.extent.quarters_interval_float import QuartersIntervalFloat
from bopp.models.v1.extent.quarters_interval_fraction import QuartersIntervalFraction
from bopp.models.v1.extent.quarters_time_float import QuartersTimeFloat
from bopp.models.v1.extent.quarters_time_fraction import QuartersTimeFraction
from bopp.registries.v1 import COMPLEX_FIELDS_REGISTRY, EXTENT_TYPE_REGISTRY

EXTENTS = [
    ("quarters_time.float", QuartersTimeFloat, {"quarter": [-1.5, 0.0, 2.25]}),
    (
        "quarters_time.fraction",
        QuartersTimeFraction,
        {"quarter": [[-1, 2], [0, 1], [9, 4]]},
    ),
    (
        "quarters_interval.float",
        QuartersIntervalFloat,
        {"quarter": [-1.5, 0.0, 2.25], "duration": [0.0, 0.5, 1.25]},
    ),
    (
        "quarters_interval.fraction",
        QuartersIntervalFraction,
        {"quarter": [[-1, 2], [0, 1], [9, 4]], "duration": [[0, 1], [1, 2], [5, 4]]},
    ),
]


INVALID_QUARTERS = [
    ("quarters_time.float", QuartersTimeFloat, {"quarter": [[1, 2]]}, "fraction"),
    (
        "quarters_interval.float",
        QuartersIntervalFloat,
        {"quarter": [[1, 2]], "duration": [0.5]},
        "fraction",
    ),
    (
        "quarters_time.fraction",
        QuartersTimeFraction,
        {"quarter": [[1, 0]]},
        "zero_denominator",
    ),
    ("quarters_time.fraction", QuartersTimeFraction, {"quarter": [0.5]}, "float"),
    (
        "quarters_interval.fraction",
        QuartersIntervalFraction,
        {"quarter": [[1, 0]], "duration": [[1, 2]]},
        "zero_denominator",
    ),
    (
        "quarters_interval.fraction",
        QuartersIntervalFraction,
        {"quarter": [0.5], "duration": [[1, 2]]},
        "float",
    ),
]


@pytest.mark.parametrize(("tag", "cls"), [(tag, cls) for tag, cls, _ in EXTENTS])
def test_extent_registry(tag, cls):
    """Each quarter extent tag resolves to its generated model class."""
    assert EXTENT_TYPE_REGISTRY[tag] is cls


@pytest.mark.parametrize(("tag", "cls", "columns"), EXTENTS)
def test_decode_valid_quarters(tag, cls, columns):
    """Quarter extents accept signed positions and nonnegative durations."""
    decoded = msgspec.json.decode(
        msgspec.json.encode({"extent_type": tag, **columns}), type=cls
    )
    for name, expected in columns.items():
        actual = getattr(decoded, name)
        if tag.endswith("fraction"):
            assert all(isinstance(value, tuple) for value in actual)
            actual = [list(value) for value in actual]
        assert actual == expected


@pytest.mark.parametrize(
    ("tag", "cls", "columns", "reason"),
    INVALID_QUARTERS,
    ids=[f"{tag}-rejects-{reason}" for tag, _, _, reason in INVALID_QUARTERS],
)
def test_decode_invalid_quarters(tag, cls, columns, reason):
    """Quarter extents enforce their numeric form and fraction denominators."""
    with pytest.raises(msgspec.ValidationError, match=r"\$\.quarter"):
        msgspec.json.decode(
            msgspec.json.encode({"extent_type": tag, **columns}),
            type=cls,
        )


@pytest.mark.parametrize(
    ("tag", "cls", "quarter", "duration"),
    [
        ("quarters_interval.float", QuartersIntervalFloat, [0.0], [-0.5]),
        (
            "quarters_interval.fraction",
            QuartersIntervalFraction,
            [[0, 1]],
            [[-1, 2]],
        ),
    ],
)
def test_decode_negative_duration(tag, cls, quarter, duration):
    """Quarter intervals reject negative durations in both numeric forms."""
    with pytest.raises(msgspec.ValidationError, match=r"\$\.duration"):
        msgspec.json.decode(
            msgspec.json.encode(
                {"extent_type": tag, "quarter": quarter, "duration": duration}
            ),
            type=cls,
        )


@pytest.mark.parametrize(("tag", "cls", "columns"), EXTENTS)
@pytest.mark.parametrize(
    ("save", "load", "suffix", "keeps_id"),
    [
        (save_bopp_json, load_bopp_json, "json", True),
        (save_bopp_msgpack, load_bopp_msgpack, "msgpack", True),
        (save_bopp_csv, load_bopp_csv, "csv", False),
    ],
)
def test_annotation_roundtrip(
    tmp_path, tag, cls, columns, save, load, suffix, keeps_id
):
    """Validated annotations preserve columns in all three storage formats."""
    values = ["pickup", "start", "later"]
    annotation = bopp.create(
        media_id="test:quarters",
        payload_kind="tag_open",
        extent_kind=tag,
        value=values,
        **columns,
    )
    assert bopp.validate(annotation)
    path = tmp_path / f"{tag}.{suffix}"
    save(annotation, path)
    loaded = load(path)
    assert bopp.validate(loaded)
    assert isinstance(loaded.extent, cls)
    assert loaded.media_id == "test:quarters"
    assert loaded.payload.value == values
    for name, expected in columns.items():
        actual = getattr(loaded.extent, name)
        if tag.endswith("fraction"):
            actual = [list(value) for value in actual]
        assert actual == expected
    if keeps_id:
        assert loaded.id == annotation.id


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("quarters_time.float", None),
        ("quarters_time.fraction", ["quarter"]),
        ("quarters_interval.float", None),
        ("quarters_interval.fraction", ["quarter", "duration"]),
    ],
)
def test_complex_fields_registry(tag, expected):
    """Only fractional quarter extent columns require complex serialization."""
    registry = COMPLEX_FIELDS_REGISTRY["extent_type"]
    if expected is not None:
        assert registry[tag] == expected
    else:
        assert tag not in registry
