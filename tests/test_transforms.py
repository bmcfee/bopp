import copy
import math
from decimal import Decimal
from fractions import Fraction

import msgspec
import numpy as np
import pytest

import bopp
from bopp.core import create
from bopp.exceptions import BoppArgumentError, BoppValidationError
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.confidence.likelihood import LikelihoodConfidence
from bopp.models.v1.extent.time import Time
from bopp.models.v1.extent.time_frequency_box import TimeFrequencyBoxExtent
from bopp.models.v1.grouping import Grouping
from bopp.models.v1.metadata.derived import DerivedAnnotationMetadata
from bopp.models.v1.metadata.human import HumanAnnotationMetadata
from bopp.models.v1.payload.mood_thayer import MoodThayerPayload
from bopp.models.v1.payload.tag import TagPayload
from bopp.transforms import (
    AXIS_CONFIGS,
    DEFAULT_TARGET_FIELDS,
    FRACTION_AXES,
    FilterRecord,
    _derive_parents_and_sandbox,
    _get_facet_list_fields,
    _rebuild_annotation,
    _resample_groupings,
    _resample_struct,
    _subset_groupings,
    _subset_struct_lists,
    filter_by,
    shift,
    sort,
    to_pixels,
    to_times,
    to_uniform_times,
    trim,
    union,
)


def test_get_facet_list_fields_empty():
    assert _get_facet_list_fields(None) == {}
    assert _get_facet_list_fields(msgspec.UNSET) == {}


def test_derive_parents_and_sandbox_helpers():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")

    # When annotation has an ID and existing parents
    ann = Annotation(
        id="ann_1",
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        parents=["parent_0"],
        sandbox={"custom": [1, 2]},
        extent=msgspec.UNSET,
        payload=TagPayload(value=["pop"]),
    )
    parents, sandbox = _derive_parents_and_sandbox(ann, "Testing")
    assert parents == ["parent_0", "ann_1"]
    assert sandbox == {"custom": [1, 2]}
    assert sandbox is not ann.sandbox

    # When annotation has no ID and UNSET parents
    ann_no_id = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagPayload(value=["pop"]),
    )
    with pytest.warns(UserWarning, match="Testing an annotation with no ID"):
        parents_no_id, sandbox_no_id = _derive_parents_and_sandbox(ann_no_id, "Testing")
    assert parents_no_id == []
    assert sandbox_no_id is msgspec.UNSET


def test_derive_parents_when_existing_parents_is_none():
    """Verify _derive_parents_and_sandbox when parents attribute is explicitly None."""
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        id="ann_none_parents",
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        parents=None,
        payload=TagPayload(value=["test"]),
    )
    parents, sandbox = _derive_parents_and_sandbox(ann, "Testing")
    assert parents == ["ann_none_parents"]
    assert sandbox is msgspec.UNSET


def test_subset_struct_lists_helper():
    assert _subset_struct_lists(None, [0, 1]) is None
    assert _subset_struct_lists(msgspec.UNSET, [0, 1]) is msgspec.UNSET

    class SampleStruct(msgspec.Struct):
        tags: list[str]
        scores: list[float]
        extra_scalar: int = 42

    sample = SampleStruct(tags=["a", "b", "c"], scores=[1.0, 2.0, 3.0])
    subsetted = _subset_struct_lists(sample, [0, 2])
    assert subsetted.tags == ["a", "c"]
    assert subsetted.scores == [1.0, 3.0]
    assert subsetted.extra_scalar == 42

    # With overrides
    overridden = _subset_struct_lists(sample, [1], overrides={"tags": ["custom"]})
    assert overridden.tags == ["custom"]
    assert overridden.scores == [2.0]


def test_subset_groupings_helper():
    assert _subset_groupings(None, [0, 1]) is None
    assert _subset_groupings(msgspec.UNSET, [0, 1]) is msgspec.UNSET

    g1 = Grouping(
        name="pattern",
        ids=[1, 2, 3, 2],
        labels={"A": 1, "B": 2, "C": 3},
        disjoint=False,
        exhaustive=False,
    )
    g2 = Grouping(name="source", ids=[0, 1, 0, 1], disjoint=False, exhaustive=False)
    subsetted = _subset_groupings([g1, g2], [1, 3])
    assert len(subsetted) == 2
    assert subsetted[0].name == "pattern"
    assert subsetted[0].ids == [2, 2]
    assert subsetted[0].labels == {"A": 1, "B": 2, "C": 3}
    assert subsetted[1].name == "source"
    assert subsetted[1].ids == [1, 1]


def test_resample_struct_helper():
    assert _resample_struct(None, {}, [], {}) is None
    assert _resample_struct(msgspec.UNSET, {}, [], {}) is msgspec.UNSET

    payload = TagPayload(value=["apple", "banana"])
    cols = _get_facet_list_fields(payload)
    resampled = _resample_struct(
        payload, cols, [1, None, 0], fill_map={"value": "empty"}
    )
    assert resampled.value == ["banana", "empty", "apple"]


def test_resample_groupings_helper():
    assert _resample_groupings(None, [], {}) is None
    assert _resample_groupings(msgspec.UNSET, [], {}) is msgspec.UNSET

    g = Grouping(
        name="section",
        ids=[0, 1],
        labels={"intro": 0, "verse": 1},
        disjoint=False,
        exhaustive=False,
    )
    resampled = _resample_groupings([g], [0, None, 1], fill_map={"section": -1})
    assert len(resampled) == 1
    assert resampled[0].ids == [0, -1, 1]
    assert resampled[0].name == "section"
    assert resampled[0].labels == {"intro": 0, "verse": 1}


def test_rebuild_annotation_helper():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    g = Grouping(name="layer", ids=[1], disjoint=False, exhaustive=False)
    ann = Annotation(
        id="old_id",
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Time(time=[1.0], disjoint=False, exhaustive=False),
        payload=TagPayload(value=["test"]),
        groupings=[g],
    )
    rebuilt = _rebuild_annotation(
        ann,
        parents=["old_id"],
        sandbox={"foo": "bar"},
        payload=TagPayload(value=["new_test"]),
        extent=Time(time=[2.0], disjoint=False, exhaustive=False),
        groupings=[Grouping(name="layer", ids=[2], disjoint=False, exhaustive=False)],
    )
    assert rebuilt.id != "old_id"
    assert rebuilt.parents == ["old_id"]
    assert rebuilt.sandbox == {"foo": "bar"}
    assert rebuilt.payload.value == ["new_test"]
    assert rebuilt.extent.time == [2.0]
    assert rebuilt.groupings[0].ids == [2]


def test_trim_invalid_arguments():
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0],
        value=["rock", "pop"],
    )

    # No start or end provided
    with pytest.raises(BoppArgumentError, match="At least one of 'start' or 'end'"):
        trim(ann)

    # start > end
    with pytest.raises(BoppArgumentError, match="start .* must be <= end"):
        trim(ann, start=10.0, end=5.0)

    # reset=True with start=None
    with pytest.raises(BoppArgumentError, match="reset=True requires 'start'"):
        trim(ann, end=5.0, reset=True)

    # Unsupported target_field for valid extent_tag
    with pytest.raises(BoppArgumentError, match="Unsupported target_field 'invalid'"):
        trim(ann, start=1.0, end=2.0, target_field="invalid")


def test_trim_argument_error_order():
    tm = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="time",
        time=[0.0, 1.0],
        value=["a", "b"],
    )
    pb = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0],
        y=[0.0],
        width=[1.0],
        height=[1.0],
        value=["a"],
    )
    with pytest.raises(BoppArgumentError, match="At least one of"):
        trim(tm, target_field="x")
    with pytest.raises(BoppArgumentError, match="Unsupported target_field"):
        trim(tm, start=2, end=1, target_field="x")
    with pytest.raises(BoppArgumentError, match="must be <= end"):
        trim(tm, start=2, end=1)
    with pytest.raises(BoppArgumentError, match="reset=True requires"):
        trim(tm, end=1, reset=True, target_field="zz")
    with pytest.raises(BoppArgumentError, match="At least one of"):
        trim(pb)
    with pytest.raises(BoppArgumentError, match="does not have a default target field"):
        trim(pb, start=0)
    with pytest.raises(BoppArgumentError, match="Unsupported target_field"):
        trim(tm, start=0, target_field="x")


def test_trim_warning_no_id():
    metadata = HumanAnnotationMetadata(annotator_id="user_1", tool="manual")
    ann = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagPayload(value=["pop"]),
    )

    with pytest.warns(UserWarning, match="Trimming an annotation with no ID"):
        result = trim(ann, start=1.0, end=5.0)

    assert result.parents == []
    assert isinstance(result.metadata, DerivedAnnotationMetadata)
    assert result.metadata.transform == "trim"


def test_trim_invalid_extent_tag():
    class UntaggedExtent(msgspec.Struct):
        time: list[float]

    ann = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=UntaggedExtent(time=[1.0, 2.0]),
        payload=TagPayload(value=["a", "b"]),
    )
    bopp.ensure_annotation_id(ann)

    with pytest.raises(BoppArgumentError, match="has no valid schema tag"):
        trim(ann, start=1.0, end=2.0)


def test_trim_no_extent():
    metadata = HumanAnnotationMetadata(annotator_id="user_1", tool="manual")
    ann = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagPayload(value=["pop"]),
    )
    bopp.ensure_annotation_id(ann)

    result = trim(ann, start=1.0, end=5.0)
    assert result.extent is msgspec.UNSET
    assert result.payload.value == ["pop"]
    assert result is not ann
    assert result.parents == [ann.id]
    assert isinstance(result.metadata, DerivedAnnotationMetadata)
    assert result.metadata.transform == "trim"


def test_trim_time_extent():
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        time=[1.0, 3.0, 5.0, 8.0],
        value=["a", "b", "c", "d"],
        confidence=[0.1, 0.3, 0.5, 0.8],
    )

    trimmed = trim(ann, start=2.0, end=6.0, reset=True)
    assert trimmed.extent.time == [1.0, 3.0]  # 3.0 - 2.0, 5.0 - 2.0
    assert trimmed.payload.value == ["b", "c"]
    assert trimmed.confidence.confidence == [0.3, 0.5]
    assert isinstance(trimmed.metadata, DerivedAnnotationMetadata)
    assert trimmed.metadata.transform == "trim"
    assert trimmed.metadata.parameters["start"] == 2.0
    assert trimmed.metadata.parameters["end"] == 6.0
    assert trimmed.metadata.parameters["reset"] is True


def test_trim_with_groupings():
    g1 = Grouping(
        name="pattern",
        ids=[0, 1, 1, 2],
        labels={"intro": 0, "verse": 1, "outro": 2},
        disjoint=False,
        exhaustive=False,
    )
    g2 = Grouping(name="stream", ids=[10, 11, 12, 13], disjoint=False, exhaustive=False)
    ann = Annotation(
        media_id="track:test_media",
        bopp_version="1.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=Time(time=[1.0, 3.0, 5.0, 8.0], disjoint=False, exhaustive=False),
        payload=TagPayload(value=["a", "b", "c", "d"]),
        confidence=LikelihoodConfidence(confidence=[0.1, 0.3, 0.5, 0.8]),
        groupings=[g1, g2],
    )
    bopp.ensure_annotation_id(ann)

    trimmed = trim(ann, start=2.0, end=6.0, reset=True)
    assert trimmed.extent.time == [1.0, 3.0]
    assert trimmed.payload.value == ["b", "c"]
    assert len(trimmed.groupings) == 2
    assert trimmed.groupings[0].name == "pattern"
    assert trimmed.groupings[0].ids == [1, 1]
    assert trimmed.groupings[0].labels == {"intro": 0, "verse": 1, "outro": 2}
    assert trimmed.groupings[1].name == "stream"
    assert trimmed.groupings[1].ids == [11, 12]
    assert bopp.validate(trimmed)

    # Empty trim
    empty = trim(ann, start=20.0, end=30.0)
    assert empty.extent.time == []
    assert empty.payload.value == []
    assert empty.groupings[0].ids == []
    assert empty.groupings[1].ids == []
    assert bopp.validate(empty)


def test_trim_numeric_bounds_pass_through():
    tm = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="time",
        time=[0.0, 1.0, 2.0],
        value=list("abc"),
    )
    result = trim(tm, start=1, end=math.inf)
    assert result.extent.time == [1.0, 2.0]
    assert result.metadata.parameters["end"] == math.inf

    result = trim(tm, start=-math.inf, end=1)
    assert result.extent.time == [0.0, 1.0]
    assert result.metadata.parameters["start"] == -math.inf

    with pytest.raises(BoppArgumentError, match="Trim bound must be an int or float"):
        trim(tm, start=Decimal("0.5"))


def test_trim_no_extent_infinite_bound():
    ann = Annotation(
        media_id="track:1",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=msgspec.UNSET,
        payload=TagPayload(value=["pop"]),
    )
    bopp.ensure_annotation_id(ann)

    result = trim(ann, start=0, end=math.inf)
    assert result.extent is msgspec.UNSET
    assert result.payload == ann.payload
    assert result.metadata.parameters["end"] == math.inf


def test_trim_point_extent_open_ended_bounds():
    """Verify trim on point extent with only start, only end, and check boundary conditions."""
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0, 3.0, 4.0, 5.0],
        value=["a", "b", "c", "d", "e"],
    )

    # Only start bound
    res_start = trim(ann, start=3.0)
    assert res_start.extent.time == [3.0, 4.0, 5.0]
    assert res_start.payload.value == ["c", "d", "e"]

    # Only end bound
    res_end = trim(ann, end=3.0)
    assert res_end.extent.time == [1.0, 2.0, 3.0]
    assert res_end.payload.value == ["a", "b", "c"]


def test_trim_time_interval_non_strict():
    # Intervals: [0, 2], [2, 6], [5, 9]
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time_interval",
        confidence_type="likelihood",
        time=[0.0, 2.0, 5.0],
        duration=[2.0, 4.0, 4.0],
        value=["first", "second", "third"],
        confidence=[0.2, 0.4, 0.6],
    )

    # Trim to [1.0, 6.0] non-strict
    trimmed = trim(ann, start=1.0, end=6.0, strict=False, reset=False)
    assert trimmed.extent.time == [1.0, 2.0, 5.0]
    assert trimmed.extent.duration == [1.0, 4.0, 1.0]
    assert trimmed.payload.value == ["first", "second", "third"]
    assert trimmed.confidence.confidence == [0.2, 0.4, 0.6]

    # With reset=True
    trimmed_reset = trim(ann, start=1.0, end=6.0, strict=False, reset=True)
    assert trimmed_reset.extent.time == [0.0, 1.0, 4.0]
    assert trimmed_reset.extent.duration == [1.0, 4.0, 1.0]


def test_trim_time_interval_strict():
    # Intervals: [0, 2], [2, 6], [5, 9]
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 2.0, 5.0],
        duration=[2.0, 4.0, 4.0],
        value=["first", "second", "third"],
    )

    # Trim to [1.0, 6.0] strict (only [2, 6] is strictly contained)
    trimmed = trim(ann, start=1.0, end=6.0, strict=True, reset=False)
    assert trimmed.extent.time == [2.0]
    assert trimmed.extent.duration == [4.0]
    assert trimmed.payload.value == ["second"]


def test_trim_time_frequency_box():
    ann = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=TimeFrequencyBoxExtent(
            time=[0.0, 4.0],
            duration=[3.0, 5.0],
            freq_min=[100.0, 200.0],
            freq_max=[500.0, 800.0],
            disjoint=False,
            exhaustive=False,
        ),
        payload=TagPayload(value=["low", "high"]),
    )
    bopp.ensure_annotation_id(ann)

    trimmed = trim(ann, start=1.0, end=5.0, strict=False, reset=True)
    assert trimmed.extent.time == [0.0, 3.0]
    assert trimmed.extent.duration == [2.0, 1.0]
    assert trimmed.extent.freq_min == [100.0, 200.0]
    assert trimmed.extent.freq_max == [500.0, 800.0]
    assert trimmed.payload.value == ["low", "high"]


def test_trim_time_frequency_box_along_frequency():
    extent = TimeFrequencyBoxExtent(
        time=[0.0, 1.0, 2.0, 3.0],
        duration=[1.0, 1.0, 1.0, 1.0],
        freq_min=[100.0, 200.0, 500.0, 900.0],
        freq_max=[300.0, 400.0, 800.0, 1000.0],
        disjoint=False,
        exhaustive=False,
    )
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=extent,
        payload=TagPayload(value=["low", "mid", "high", "ultra"]),
    )
    bopp.ensure_annotation_id(ann)

    # Trim frequency axis non-strict [250..600]
    trimmed_freq = trim(
        ann, start=250.0, end=600.0, target_field="frequency", reset=True
    )
    assert trimmed_freq.extent.freq_min == [0.0, 0.0, 250.0]
    assert trimmed_freq.extent.freq_max == [50.0, 150.0, 350.0]
    assert trimmed_freq.payload.value == ["low", "mid", "high"]

    # Trim frequency axis strict (150 to 450 strictly contains [200, 400])
    trimmed_freq_strict = trim(
        ann, start=150.0, end=450.0, target_field="frequency", strict=True
    )
    assert trimmed_freq_strict.extent.freq_min == [200.0]
    assert trimmed_freq_strict.extent.freq_max == [400.0]
    assert trimmed_freq_strict.payload.value == ["mid"]


def test_trim_min_max_strict_and_non_strict():
    extent = TimeFrequencyBoxExtent(
        time=[0.0, 0.0, 0.0, 0.0],
        duration=[1.0, 1.0, 1.0, 1.0],
        freq_min=[50.0, 200.0, 500.0, 700.0],
        freq_max=[100.0, 400.0, 800.0, 900.0],
        disjoint=False,
        exhaustive=False,
    )
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=extent,
        payload=TagPayload(value=["low", "mid", "high", "ultra"]),
    )
    bopp.ensure_annotation_id(ann)

    # Non-strict trim with start and end
    trimmed_non_strict = trim(
        ann, start=150.0, end=600.0, target_field="frequency", strict=False
    )
    assert trimmed_non_strict.extent.freq_min == [200.0, 500.0]
    assert trimmed_non_strict.extent.freq_max == [400.0, 600.0]
    assert trimmed_non_strict.payload.value == ["mid", "high"]

    # Non-strict trim with only start (tests start is not None, end is None branch)
    trimmed_only_start_ns = trim(
        ann, start=300.0, target_field="frequency", strict=False
    )
    assert trimmed_only_start_ns.extent.freq_min == [300.0, 500.0, 700.0]
    assert trimmed_only_start_ns.extent.freq_max == [400.0, 800.0, 900.0]

    # Non-strict trim with only end (tests start is None, end is not None branch)
    trimmed_only_end_ns = trim(ann, end=450.0, target_field="frequency", strict=False)
    assert trimmed_only_end_ns.extent.freq_min == [50.0, 200.0]
    assert trimmed_only_end_ns.extent.freq_max == [100.0, 400.0]

    # Strict trim with start and end
    trimmed_strict = trim(
        ann, start=150.0, end=600.0, target_field="frequency", strict=True
    )
    assert trimmed_strict.extent.freq_min == [200.0]
    assert trimmed_strict.extent.freq_max == [400.0]
    assert trimmed_strict.payload.value == ["mid"]

    # Strict trim with only start
    trimmed_only_start_s = trim(ann, start=150.0, target_field="frequency", strict=True)
    assert trimmed_only_start_s.extent.freq_min == [200.0, 500.0, 700.0]

    # Strict trim with only end
    trimmed_only_end_s = trim(ann, end=600.0, target_field="frequency", strict=True)
    assert trimmed_only_end_s.extent.freq_min == [50.0, 200.0]


def test_trim_empty_and_out_of_bounds_extents():
    """Verify that trimming empty extents or extents where all observations fall outside
    the specified range results in valid, empty array attributes and correct parent lineage.
    """
    # 1. Edge case: Extent has empty observation arrays
    extent_empty = TimeFrequencyBoxExtent(
        time=[],
        duration=[],
        freq_min=[],
        freq_max=[],
        disjoint=False,
        exhaustive=False,
    )
    ann_empty = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=extent_empty,
        payload=TagPayload(value=[]),
    )
    bopp.ensure_annotation_id(ann_empty)

    trimmed_empty = trim(
        ann_empty, start=100.0, end=200.0, target_field="frequency", strict=True
    )
    assert trimmed_empty.extent.freq_min == []
    assert trimmed_empty.extent.freq_max == []
    assert trimmed_empty.payload.value == []
    assert trimmed_empty.parents == [ann_empty.id]

    # 2. Functional case: Extent with observations completely outside [start, end]
    extent_out_of_bounds = TimeFrequencyBoxExtent(
        time=[0.0, 1.0],
        duration=[1.0, 1.0],
        freq_min=[10.0, 20.0],
        freq_max=[50.0, 60.0],
        disjoint=False,
        exhaustive=False,
    )
    ann_oob = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=extent_out_of_bounds,
        payload=TagPayload(value=["low1", "low2"]),
    )
    bopp.ensure_annotation_id(ann_oob)

    # Trim frequency axis [100.0, 200.0] - all observations should be filtered out
    trimmed_oob = trim(
        ann_oob, start=100.0, end=200.0, target_field="frequency", strict=False
    )
    assert trimmed_oob.extent.freq_min == []
    assert trimmed_oob.extent.freq_max == []
    assert trimmed_oob.payload.value == []
    assert trimmed_oob.parents == [ann_oob.id]


def test_trim_pixel_box():
    ann = create(
        media_id="test_image",
        payload_type="tag",
        extent_type="pixel_box",
        x=[10.0, 50.0, 100.0],
        width=[20.0, 30.0, 40.0],
        y=[100.0, 200.0, 300.0],
        height=[50.0, 50.0, 50.0],
        value=["obj1", "obj2", "obj3"],
    )

    # Requiring target_field for pixel_box
    with pytest.raises(BoppArgumentError, match="does not have a default target field"):
        trim(ann, start=20.0, end=70.0)

    # Trimming along x
    trimmed_x = trim(ann, start=20.0, end=70.0, target_field="x", reset=True)
    assert trimmed_x.extent.x == [0.0, 30.0]  # [30-20, 50-20]
    assert trimmed_x.extent.width == [10.0, 20.0]
    assert trimmed_x.extent.y == [100.0, 200.0]
    assert trimmed_x.payload.value == ["obj1", "obj2"]

    # Trimming along y (180.0 to 260.0 keeps obj2 [200..250])
    trimmed_y = trim(ann, start=180.0, end=260.0, target_field="y", reset=False)
    assert trimmed_y.extent.y == [200.0]
    assert trimmed_y.extent.height == [50.0]
    assert trimmed_y.payload.value == ["obj2"]


def test_trim_transitive_parents_and_immutability():
    ann1 = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 3.0, 5.0, 8.0],
        value=["a", "b", "c", "d"],
    )
    ann1_copy = copy.deepcopy(ann1)

    ann2 = trim(ann1, start=2.0, end=7.0)
    assert ann2.parents == [ann1.id]
    assert ann2.id != ann1.id

    # Verify original annotation was not mutated
    assert ann1 == ann1_copy

    ann2_copy = copy.deepcopy(ann2)
    ann3 = trim(ann2, start=2.5, end=6.0)

    # Transitive parent chain check
    assert ann3.parents == [ann1.id, ann2.id]
    assert ann3.id != ann2.id
    assert ann3.id != ann1.id

    # Verify second annotation was not mutated
    assert ann2 == ann2_copy


def test_trim_extent_without_bounds(monkeypatch):
    ann = create(
        media_id="test",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0],
        value=["a", "b"],
    )
    with monkeypatch.context() as m:
        m.setattr("bopp.transforms._get_extent_axis_bound_attrs", lambda *args: None)
        res = trim(ann, start=1.5)
        assert res.extent.time == [2.0]


def test_filter_record_attribute_access():
    rec = FilterRecord({"a": 1, "b": "hello"})
    assert rec.a == 1
    assert rec.b == "hello"
    assert rec["a"] == 1
    rec.c = True
    assert rec.c is True
    assert rec["c"] is True

    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        _ = rec.nonexistent


def test_filter_by_default_payload():
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        time=[1.0, 2.0, 3.0, 4.0],
        value=["rock", "pop", "rock", "jazz"],
        confidence=[0.5, 0.6, 0.7, 0.8],
    )

    filtered = filter_by(ann, lambda v: v == "rock")
    assert filtered.payload.value == ["rock", "rock"]
    assert filtered.extent.time == [1.0, 3.0]
    assert filtered.confidence.confidence == [0.5, 0.7]
    assert filtered.parents == [ann.id]
    assert filtered.id != ann.id
    assert isinstance(filtered.metadata, DerivedAnnotationMetadata)
    assert filtered.metadata.transform == "filter_by"
    assert filtered.metadata.parameters["facet"] == "payload"


def test_filter_by_target_on_payload():
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0, 3.0],
        value=["rock", "pop", "metal"],
    )

    filtered = filter_by(ann, lambda v: len(v) == 4, target="value")
    assert filtered.payload.value == ["rock"]
    assert filtered.extent.time == [1.0]


def test_filter_by_target_on_extent():
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        time=[1.0, 2.0, 3.0, 4.0],
        value=["a", "b", "c", "d"],
        confidence=[0.1, 0.2, 0.3, 0.4],
    )

    filtered = filter_by(ann, lambda t: t > 2.0, facet="extent", target="time")
    assert filtered.extent.time == [3.0, 4.0]
    assert filtered.payload.value == ["c", "d"]
    assert filtered.confidence.confidence == [0.3, 0.4]

    filtered_notarget = filter_by(ann, lambda t: t > 2.0, facet="extent")
    assert filtered_notarget.extent.time == [3.0, 4.0]


def test_filter_by_confidence_facet():
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        time=[1.0, 2.0, 3.0],
        value=["a", "b", "c"],
        confidence=[0.2, 0.8, 0.5],
    )

    filtered = filter_by(
        ann, lambda c: c >= 0.5, facet="confidence", target="confidence"
    )
    assert filtered.confidence.confidence == [0.8, 0.5]
    assert filtered.payload.value == ["b", "c"]
    assert filtered.extent.time == [2.0, 3.0]


def test_filter_by_groupings_facet():
    g1 = Grouping(
        name="pattern",
        ids=[0, 1, 1, 2],
        labels={"intro": 0, "verse": 1, "outro": 2},
        disjoint=False,
        exhaustive=False,
    )
    g2 = Grouping(name="track", ids=[10, 20, 20, 30], disjoint=False, exhaustive=False)
    ann = Annotation(
        media_id="track:test_media",
        bopp_version="1.0",
        metadata=HumanAnnotationMetadata(annotator_id="u1", tool="manual"),
        extent=Time(time=[1.0, 2.0, 3.0, 4.0], disjoint=False, exhaustive=False),
        payload=TagPayload(value=["a", "b", "c", "d"]),
        groupings=[g1, g2],
    )
    bopp.ensure_annotation_id(ann)

    # Filter by specific grouping target
    res_verse = filter_by(
        ann, lambda gid: gid == 1, facet="groupings", target="pattern"
    )
    assert res_verse.payload.value == ["b", "c"]
    assert res_verse.extent.time == [2.0, 3.0]
    assert res_verse.groupings[0].ids == [1, 1]
    assert res_verse.groupings[1].ids == [20, 20]
    assert bopp.validate(res_verse)

    # Filter with record access across multiple grouping layers
    res_rec = filter_by(
        ann, lambda r: r.pattern == 1 and r.track == 20, facet="groupings"
    )
    assert res_rec.payload.value == ["b", "c"]

    # Filter when only 1 grouping is present and target is None
    g_single = Grouping(
        name="pattern",
        ids=[0, 1],
        labels={"intro": 0, "verse": 1},
        disjoint=False,
        exhaustive=False,
    )
    ann_single_g = Annotation(
        media_id="track:test_media",
        bopp_version="1.0",
        metadata=HumanAnnotationMetadata(annotator_id="u1", tool="manual"),
        extent=Time(time=[1.0, 2.0], disjoint=False, exhaustive=False),
        payload=TagPayload(value=["a", "b"]),
        groupings=[g_single],
    )
    bopp.ensure_annotation_id(ann_single_g)
    res_single = filter_by(ann_single_g, lambda gid: gid == 0, facet="groupings")
    assert res_single.payload.value == ["a"]


def test_filter_by_multi_column_payload():
    # MoodThayerPayload has valence and arousal
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="gui")
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=MoodThayerPayload(valence=[0.5, -0.2, 0.8], arousal=[0.1, 0.4, -0.3]),
    )
    bopp.ensure_annotation_id(ann)

    # Filter with record
    filtered = filter_by(ann, lambda r: r.valence > 0 and r.arousal > 0)
    assert filtered.payload.valence == [0.5]
    assert filtered.payload.arousal == [0.1]

    # Explicit target on one field
    filtered_v = filter_by(ann, lambda v: v > 0, target="valence")
    assert filtered_v.payload.valence == [0.5, 0.8]
    assert filtered_v.payload.arousal == [0.1, -0.3]


def test_filter_by_facet_all():
    g = Grouping(name="pattern", ids=[1, 1, 2, 1], disjoint=False, exhaustive=False)
    ann = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="u1", tool="manual"),
        extent=Time(time=[1.0, 2.0, 3.0, 4.0], disjoint=False, exhaustive=False),
        payload=TagPayload(value=["intro", "verse", "chorus", "verse"]),
        confidence=LikelihoodConfidence(confidence=[0.9, 0.4, 0.95, 0.85]),
        groupings=[g],
    )
    bopp.ensure_annotation_id(ann)

    # Filter across facets using record (extent, payload, confidence, grouping)
    filtered = filter_by(
        ann,
        lambda r: r.confidence >= 0.8 and r.value.startswith("v") and r.pattern == 1,
        facet="all",
    )
    assert filtered.payload.value == ["verse"]
    assert filtered.extent.time == [4.0]
    assert filtered.confidence.confidence == [0.85]
    assert filtered.groupings[0].ids == [1]

    # Filter across facets specifying target on grouping name
    filtered_target_g = filter_by(
        ann,
        lambda gid: gid == 2,
        facet="all",
        target="pattern",
    )
    assert filtered_target_g.extent.time == [3.0]
    assert filtered_target_g.payload.value == ["chorus"]
    assert filtered_target_g.groupings[0].ids == [2]

    # Filter across facets specifying target on time
    filtered_target = filter_by(
        ann,
        lambda t: t <= 2.0,
        facet="all",
        target="time",
    )
    assert filtered_target.extent.time == [1.0, 2.0]
    assert filtered_target.payload.value == ["intro", "verse"]


def test_filter_by_skips_empty_facet_in_obs_counting():
    """Verify that filter_by handles annotations where extent is unset (empty dict in all_facets_cols)."""
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        id="ann_no_extent",
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagPayload(value=["one", "two", "three"]),
    )
    filtered = filter_by(ann, lambda v: v != "two")
    assert filtered.payload.value == ["one", "three"]
    assert filtered.extent is msgspec.UNSET


def test_filter_by_invalid_arguments():
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0],
        value=["a", "b"],
    )

    with pytest.raises(BoppArgumentError, match="Invalid facet 'invalid'"):
        filter_by(ann, lambda x: True, facet="invalid")  # type: ignore[arg-type]

    with pytest.raises(BoppArgumentError, match="Facet 'confidence' is not present"):
        filter_by(ann, lambda x: True, facet="confidence")

    with pytest.raises(BoppArgumentError, match="Facet 'groupings' is not present"):
        filter_by(ann, lambda x: True, facet="groupings")

    with pytest.raises(
        BoppArgumentError, match="Target field 'unknown' not found in facet 'payload'"
    ):
        filter_by(ann, lambda x: True, target="unknown")

    with pytest.raises(
        BoppArgumentError,
        match="Target field 'unknown' not found in any annotation facet",
    ):
        filter_by(ann, lambda x: True, facet="all", target="unknown")

    # Groupings target not found
    ann_with_g = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="u1", tool="manual"),
        extent=Time(time=[1.0], disjoint=True, exhaustive=False),
        payload=TagPayload(value=["a"]),
        groupings=[
            Grouping(name="valid_layer", ids=[0], disjoint=True, exhaustive=False)
        ],
    )
    bopp.ensure_annotation_id(ann_with_g)
    with pytest.raises(
        BoppArgumentError, match="Target grouping 'invalid_layer' not found"
    ):
        filter_by(ann_with_g, lambda x: True, facet="groupings", target="invalid_layer")


def test_filter_by_no_id_warning():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagPayload(value=["pop"]),
    )

    with pytest.warns(UserWarning, match="Filtering an annotation with no ID"):
        res = filter_by(ann, lambda v: True)

    assert res.parents == []
    assert isinstance(res.metadata, DerivedAnnotationMetadata)
    assert res.metadata.transform == "filter_by"


def test_filter_by_empty_or_none_remaining():
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0],
        value=["a", "b"],
    )

    filtered_none = filter_by(ann, lambda v: False)
    assert filtered_none.payload.value == []
    assert filtered_none.extent.time == []
    assert filtered_none.parents == [ann.id]

    # Now filter the already empty annotation
    filtered_again = filter_by(filtered_none, lambda v: True)
    assert filtered_again.payload.value == []
    assert filtered_again.extent.time == []
    assert filtered_again.parents == [ann.id, filtered_none.id]


def test_filter_by_sandbox_and_immutability():
    ann = create(
        media_id="test_media",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0],
        value=["a", "b"],
        sandbox={"custom": [1, 2, 3]},
    )
    ann_copy = copy.deepcopy(ann)

    filtered = filter_by(ann, lambda v: v == "b")
    assert filtered.sandbox == {"custom": [1, 2, 3]}
    assert filtered.sandbox is not ann.sandbox

    # Verify original unchanged
    assert ann == ann_copy


def test_to_times_invalid_args():
    ann = create(
        media_id="test",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0],
        value=["a", "b"],
    )

    # Neither times nor sample_rate
    with pytest.raises(
        BoppArgumentError, match="Exactly one of 'times' or 'sample_rate'"
    ):
        to_times(ann)

    # Both times and sample_rate
    with pytest.raises(
        BoppArgumentError, match="Exactly one of 'times' or 'sample_rate'"
    ):
        to_times(ann, times=[1.0], sample_rate=2.0)

    # Non-positive sample_rate
    with pytest.raises(BoppArgumentError, match="sample_rate must be positive"):
        to_times(ann, sample_rate=0)
    with pytest.raises(BoppArgumentError, match="sample_rate must be positive"):
        to_times(ann, sample_rate=-1.5)

    # Invalid method
    with pytest.raises(BoppArgumentError, match="Invalid method 'invalid'"):
        to_times(ann, times=[1.0], method="invalid")  # type: ignore[arg-type]

    # Invalid overlap
    with pytest.raises(BoppArgumentError, match="Invalid overlap strategy 'invalid'"):
        to_times(ann, times=[1.0], overlap="invalid")  # type: ignore[arg-type]


def test_to_times_incompatible_extent():
    ann = create(
        media_id="img",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0],
        y=[0.0],
        width=[10.0],
        height=[10.0],
        value=["box"],
    )
    with pytest.raises(BoppArgumentError, match="incompatible with to_times"):
        to_times(ann, times=[0.0])


def test_to_times_incompatible_extent_types():
    """Verify to_times rejects unsupported extent types like midi_tick or score_quarter."""
    ann_midi = create(
        media_id="track",
        payload_type="tag",
        extent_type="midi_tick",
        tick=[0, 480],
        value=["a", "b"],
    )
    with pytest.raises(BoppArgumentError, match="is incompatible with to_times"):
        to_times(ann_midi, times=[0.0, 1.0])

    ann_score = create(
        media_id="track",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[0, 1], [1, 1]],
        value=["a", "b"],
    )
    with pytest.raises(BoppArgumentError, match="is incompatible with to_times"):
        to_times(ann_score, times=[0.0, 1.0])


def test_to_times_from_time_interval_with_sample_rate():
    ann = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time_interval",
        confidence_type="likelihood",
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=["verse", "chorus"],
        confidence=[0.8, 0.9],
    )

    res = to_times(ann, sample_rate=1.0)
    assert res.extent.time == [0.0, 1.0, 2.0, 3.0, 4.0]
    # At t=2.0, both intervals cover [0, 2] and [2, 4]; overlap defaults to 'latest'
    assert res.payload.value == ["verse", "verse", "chorus", "chorus", "chorus"]
    assert res.confidence.confidence == [0.8, 0.8, 0.9, 0.9, 0.9]
    assert res.parents == [ann.id]
    assert res.id != ann.id
    assert isinstance(res.metadata, DerivedAnnotationMetadata)
    assert res.metadata.transform == "to_times"
    assert res.metadata.parameters["sample_rate"] == 1.0


def test_to_times_with_groupings():
    g1 = Grouping(
        name="pattern",
        ids=[1, 2],
        labels={"A": 1, "B": 2},
        disjoint=False,
        exhaustive=False,
    )
    g2 = Grouping(name="source", ids=[10, 20], disjoint=False, exhaustive=False)
    ann = Annotation(
        media_id="audio:track_1",
        bopp_version="1.0",
        metadata=HumanAnnotationMetadata(annotator_id="u1", tool="manual"),
        extent=Time(time=[1.0, 3.0], disjoint=False, exhaustive=False),
        payload=TagPayload(value=["a", "b"]),
        confidence=LikelihoodConfidence(confidence=[0.7, 0.9]),
        groupings=[g1, g2],
    )
    bopp.ensure_annotation_id(ann)

    # Resample with gap: t=0.0 before first event, so method='previous' has gap
    res = to_times(
        ann,
        times=[0.0, 1.5, 3.0],
        fill_value={
            "value": "gap_val",
            "confidence": 0.0,
            "pattern": -1,
            "source": -99,
        },
    )
    assert res.extent.time == [0.0, 1.5, 3.0]
    assert res.payload.value == ["gap_val", "a", "b"]
    assert res.groupings[0].ids == [-1, 1, 2]
    assert res.groupings[1].ids == [-99, 10, 20]
    assert bopp.validate(res)

    # Test default grouping fill value (-1)
    res_default_fill = to_times(
        ann,
        times=[0.0, 1.5],
        fill_value={"value": "gap_val"},
    )
    assert res_default_fill.groupings[0].ids == [-1, 1]
    assert res_default_fill.groupings[1].ids == [-1, 10]

    # Test invalid grouping fill value type
    with pytest.raises(BoppValidationError, match="must be an integer"):
        to_times(
            ann,
            times=[0.0],
            fill_value={"value": "gap_val", "pattern": "not_an_int"},
        )


def test_to_times_overlap_strategies():
    ann = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 1.0],
        duration=[2.0, 2.0],
        value=["first", "second"],
    )

    # Sampling at t=1.5 falls into both [0, 2] and [1, 3]
    res_latest = to_times(ann, times=[1.5], overlap="latest")
    assert res_latest.payload.value == ["second"]

    res_first = to_times(ann, times=[1.5], overlap="first")
    assert res_first.payload.value == ["first"]

    res_multi = to_times(ann, times=[1.5], overlap="multiple")
    assert res_multi.extent.time == [1.5, 1.5]
    assert res_multi.payload.value == ["first", "second"]


def test_to_times_interval_gaps_and_fill_value():
    ann = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time_interval",
        confidence_type="likelihood",
        time=[1.0],
        duration=[1.0],
        value=["vocal"],
        confidence=[0.9],
    )

    res = to_times(
        ann,
        times=[0.0, 1.5, 3.0],
        fill_value="silence",
    )
    assert res.extent.time == [0.0, 1.5, 3.0]
    assert res.payload.value == ["silence", "vocal", "silence"]
    assert res.confidence.confidence == [0.0, 0.9, 0.0]


def test_to_times_from_point_time_extent():
    ann = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 3.0],
        value=["a", "b"],
    )

    # previous lookup
    res_prev = to_times(
        ann, times=[0.5, 1.5, 3.0, 4.0], method="previous", fill_value="none"
    )
    assert res_prev.payload.value == ["none", "a", "b", "b"]

    # nearest lookup
    res_near = to_times(ann, times=[0.5, 1.9, 2.1, 4.0], method="nearest")
    assert res_near.payload.value == ["a", "a", "b", "b"]


def test_to_times_global_annotation():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagPayload(value=["rock"]),
    )
    bopp.ensure_annotation_id(ann)

    # sample_rate without extent raises error
    with pytest.raises(BoppArgumentError, match="sample_rate requires an extent"):
        to_times(ann, sample_rate=1.0)

    # explicit times broadcasts
    res = to_times(ann, times=[0.0, 1.0, 2.0])
    assert res.extent.time == [0.0, 1.0, 2.0]
    assert res.payload.value == ["rock", "rock", "rock"]


def test_to_times_time_frequency_box():
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="u1", tool="manual"),
        extent=TimeFrequencyBoxExtent(
            time=[0.0, 2.0],
            duration=[2.0, 2.0],
            freq_min=[100.0, 200.0],
            freq_max=[500.0, 600.0],
            disjoint=False,
            exhaustive=False,
        ),
        payload=TagPayload(value=["low", "high"]),
    )
    bopp.ensure_annotation_id(ann)

    res = to_times(ann, times=[1.0, 3.0])
    assert res.extent.time == [1.0, 3.0]
    assert res.payload.value == ["low", "high"]


def test_to_times_fill_value_dict_with_value_fallback_and_confidence():
    """Verify fill_value dict with 'value' key fallback for single-column payload and confidence fill."""
    ann = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time_interval",
        confidence_type="likelihood",
        time=[1.0],
        duration=[1.0],
        value=["solo"],
        confidence=[0.9],
    )
    res = to_times(
        ann,
        times=[0.0, 1.5, 3.0],
        fill_value={"value": "silent", "confidence": 0.05},
    )
    assert res.extent.time == [0.0, 1.5, 3.0]
    assert res.payload.value == ["silent", "solo", "silent"]
    assert res.confidence.confidence == [0.05, 0.9, 0.05]


def test_to_times_fill_value_dict_missing_required_key():
    """Verify to_times raises BoppArgumentError when a dict fill_value omits required payload fields."""
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Time(time=[1.0], disjoint=False, exhaustive=False),
        payload=MoodThayerPayload(valence=[0.5], arousal=[0.2]),
    )
    bopp.ensure_annotation_id(ann)

    with pytest.raises(BoppArgumentError, match="Missing fill value for payload field"):
        to_times(ann, times=[0.0, 1.0], fill_value={"valence": 0.0})


def test_to_times_multi_column_payload_dict_fill():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=TimeFrequencyBoxExtent(
            time=[1.0],
            duration=[1.0],
            freq_min=[100.0],
            freq_max=[200.0],
            disjoint=False,
            exhaustive=False,
        ),
        payload=MoodThayerPayload(valence=[0.5], arousal=[0.8]),
    )
    bopp.ensure_annotation_id(ann)

    # Scalar fill value for multi-column payload raises error
    with pytest.raises(BoppArgumentError, match="requires fill_value to be a dict"):
        to_times(ann, times=[0.0], fill_value=0.0)

    # Incomplete dict raises error
    with pytest.raises(
        BoppArgumentError, match="Missing fill value for payload field 'arousal'"
    ):
        to_times(ann, times=[0.0], fill_value={"valence": 0.0})

    # Proper dict fill
    res = to_times(ann, times=[0.0, 1.5], fill_value={"valence": 0.0, "arousal": 0.0})
    assert res.extent.time == [0.0, 1.5]
    assert res.payload.valence == [0.0, 0.5]
    assert res.payload.arousal == [0.0, 0.8]


def test_to_times_multi_column_payload_none_fill_value():
    """Verify multi-column payload with fill_value=None successfully fills missing positions with None."""

    class OptionalFieldPayload(
        msgspec.Struct, tag_field="payload_type", tag="opt_payload"
    ):
        label: list[str | None]
        comment: list[str | None]

    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Time(time=[1.0], disjoint=False, exhaustive=False),
        payload=OptionalFieldPayload(label=["test"], comment=["note"]),
    )
    bopp.ensure_annotation_id(ann)

    res = to_times(ann, times=[0.0, 1.0], method="previous", fill_value=None)
    assert res.extent.time == [0.0, 1.0]
    assert res.payload.label == [None, "test"]
    assert res.payload.comment == [None, "note"]


def test_to_times_sample_rate_grid_boundary_pop():
    """Verify target_times grid trimming when step multiple lands precisely on t_max."""
    ann = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time",
        time=[0.0, 1.0],
        value=["start", "end"],
    )
    # sample_rate=1.0 over [0.0, 1.0] creates points [0.0, 1.0]
    res = to_times(ann, sample_rate=1.0)
    assert res.extent.time == [0.0, 1.0]


def test_to_times_fill_value_type_validation_error():
    ann = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.0],
        duration=[1.0],
        value=["tag"],
    )

    # tag requires list[str], so passing an invalid non-string (int) causes validation error
    with pytest.raises(BoppValidationError, match="Invalid fill value"):
        to_times(ann, times=[0.0], fill_value=123)  # type: ignore[arg-type]


def test_to_times_empty_annotation_and_empty_times():
    ann = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time",
        time=[],
        value=[],
    )

    # Empty times
    res_empty_times = to_times(ann, times=[])
    assert res_empty_times.extent.time == []
    assert res_empty_times.payload.value == []

    # Sampling empty annotation with times
    res_sampled = to_times(ann, times=[1.0, 2.0], fill_value="fill")
    assert res_sampled.extent.time == [1.0, 2.0]
    assert res_sampled.payload.value == ["fill", "fill"]


def test_to_times_empty_annotation_with_confidence_and_fill():
    """Verify sampling an empty annotation that has both payload and confidence facets."""
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Time(time=[], disjoint=False, exhaustive=False),
        payload=TagPayload(value=[]),
        confidence=LikelihoodConfidence(confidence=[]),
    )
    bopp.ensure_annotation_id(ann)

    res = to_times(ann, times=[0.0, 1.0], fill_value="silence")
    assert res.extent.time == [0.0, 1.0]
    assert res.payload.value == ["silence", "silence"]
    assert res.confidence.confidence == [0.0, 0.0]


def test_to_times_empty_annotation_confidence_fill_validation_failure():
    """Verify validation error when confidence fill value does not match confidence struct types."""
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Time(time=[], disjoint=False, exhaustive=False),
        payload=TagPayload(value=[]),
        confidence=LikelihoodConfidence(confidence=[]),
    )
    bopp.ensure_annotation_id(ann)

    with pytest.raises(
        BoppValidationError, match="Invalid fill value for confidence struct"
    ):
        to_times(
            ann,
            times=[0.0, 1.0],
            fill_value={"value": "silence", "confidence": "not_a_float"},
        )


def test_to_times_point_extent_previous_with_gaps_before_first_observation():
    """Verify that queries before the first timestamp in method='previous' produce gaps and trigger fill."""
    ann = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time",
        time=[5.0, 10.0],
        value=["a", "b"],
    )
    res = to_times(ann, times=[1.0, 5.0, 7.0], method="previous", fill_value="gap")
    assert res.extent.time == [1.0, 5.0, 7.0]
    assert res.payload.value == ["gap", "a", "a"]


def test_to_times_warning_no_id_and_immutability():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagPayload(value=["v"]),
    )

    with pytest.warns(UserWarning, match="Converting an annotation with no ID"):
        res = to_times(ann, times=[0.0])

    assert res.parents == []

    # Verify immutability
    ann_with_id = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["val"],
        sandbox={"key": 1},
    )
    ann_copy = copy.deepcopy(ann_with_id)
    res_immut = to_times(ann_with_id, times=[1.0])
    assert ann_with_id == ann_copy
    assert res_immut.sandbox is not ann_with_id.sandbox


def test_trim_fraction_points():
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[-1, 1], [-1, 2], [0, 1], [2, 4], [1, 1]],
        value=list("abcde"),
    )
    original = copy.deepcopy(ann)

    result = trim(ann, start=[-2, 4], end=Fraction(1, 2))
    assert [list(q) for q in result.extent.quarter] == [[-1, 2], [0, 1], [1, 2]]
    assert all(isinstance(q, list) for q in result.extent.quarter)
    assert result.payload.value == list("bcd")
    assert result.metadata.parameters["start"] == [-1, 2]
    assert result.metadata.parameters["end"] == [1, 2]

    shifted = trim(ann, start=Fraction(-1, 2), end=(1, 2), reset=True)
    assert [list(q) for q in shifted.extent.quarter] == [[0, 1], [1, 2], [1, 1]]
    assert shifted.payload.value == list("bcd")

    only_start = trim(ann, start=(1, 2))
    assert [list(q) for q in only_start.extent.quarter] == [[1, 2], [1, 1]]
    assert only_start.payload.value == list("de")
    only_end = trim(ann, end=0)
    assert [list(q) for q in only_end.extent.quarter] == [[-1, 1], [-1, 2], [0, 1]]
    assert only_end.payload.value == list("abc")
    assert ann == original


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_trim_fraction_intervals(reset, strict):
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[-1, 2], [0, 1], [2, 4], [1, 1], [3, 2]],
        duration=[[1, 1], [2, 2], [1, 2], [1, 1], [1, 1]],
        value=list("abcde"),
    )
    original = copy.deepcopy(ann)
    result = trim(ann, start=Fraction(1, 2), end=[3, 2], strict=strict, reset=reset)
    if strict:
        expected_quarters = [[0, 1]] if reset else [[1, 2]]
        expected_durations = [[1, 2]]
        expected_values = ["c"]
    else:
        expected_quarters = (
            [[0, 1], [0, 1], [1, 2]] if reset else [[1, 2], [1, 2], [1, 1]]
        )
        expected_durations = [[1, 2], [1, 2], [1, 2]]
        expected_values = list("bcd")
    assert [list(q) for q in result.extent.quarter] == expected_quarters
    assert [list(d) for d in result.extent.duration] == expected_durations
    assert result.payload.value == expected_values
    assert bopp.validate(result)
    assert ann == original

    empty = trim(ann, start=[5, 1], strict=strict, reset=reset)
    assert empty.extent.quarter == []
    assert empty.extent.duration == []
    assert empty.payload.value == []


@pytest.mark.parametrize(
    "strict, expected_values",
    [(False, ["inside"]), (True, ["left", "inside", "right"])],
)
def test_trim_fraction_interval_boundaries(strict, expected_values):
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[1, 2], [1, 2], [1, 1]],
        duration=[[0, 1], [1, 2], [0, 1]],
        value=["left", "inside", "right"],
    )
    result = trim(ann, start=(1, 2), end=1, strict=strict)
    assert result.payload.value == expected_values
    assert bopp.validate(result)


@pytest.mark.parametrize("bound", [0.5, Fraction(1, 2), [1, 2], (1, 2)])
@pytest.mark.parametrize("decoded", [False, True])
@pytest.mark.parametrize("extent_type", ["score_quarter", "score_interval"])
def test_trim_fraction_bound_forms_and_roundtrip(bound, decoded, extent_type):
    kwargs = {"duration": [[1, 2]] * 4} if extent_type == "score_interval" else {}
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type=extent_type,
        quarter=[[-1, 2], [0, 1], [2, 4], [1, 1]],
        value=list("abcd"),
        **kwargs,
    )
    if decoded:
        ann = msgspec.json.decode(msgspec.json.encode(ann), type=type(ann))
        assert isinstance(ann.extent.quarter[0], tuple)

    result = trim(ann, start=bound, end=Fraction(3, 2), reset=True)
    assert [list(q) for q in result.extent.quarter] == [[0, 1], [1, 2]]
    if kwargs:
        assert [list(d) for d in result.extent.duration] == [[1, 2], [1, 2]]
    assert result.payload.value == list("cd")
    assert result.metadata.parameters["start"] == (
        0.5 if isinstance(bound, float) else [1, 2]
    )
    assert result.metadata.parameters["end"] == [3, 2]
    assert not any(isinstance(v, Fraction) for v in result.metadata.parameters.values())
    assert bopp.validate(result)
    roundtrip = msgspec.json.decode(msgspec.json.encode(result), type=type(result))
    assert roundtrip.id == result.id
    assert [list(q) for q in roundtrip.extent.quarter] == [[0, 1], [1, 2]]
    assert roundtrip.payload.value == result.payload.value
    assert roundtrip.metadata.parameters == result.metadata.parameters
    assert bopp.validate(roundtrip)


def test_trim_fraction_exact_float_bound():
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1]],
        duration=[[1, 1]],
        value=["a"],
    )
    result = trim(ann, start=0.1, end=1, reset=True)
    expected_duration = Fraction(1) - Fraction(0.1)
    assert [list(q) for q in result.extent.quarter] == [[0, 1]]
    assert [list(d) for d in result.extent.duration] == [
        [expected_duration.numerator, expected_duration.denominator]
    ]
    assert result.metadata.parameters["start"] == 0.1
    assert result.metadata.parameters["end"] == 1
    assert isinstance(result.metadata.parameters["end"], int)


@pytest.mark.parametrize(
    "bound", [[1, 0], [1, -2], [1, 2, 3], [], [1.0, 2], [1, "2"], [1, True], [True, 2]]
)
@pytest.mark.parametrize("name", ["start", "end"])
def test_trim_fraction_malformed_bounds(bound, name):
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[0, 1]],
        value=["a"],
    )
    with pytest.raises(
        BoppArgumentError, match="Fraction bound must be a pair of ints"
    ):
        trim(ann, **{name: bound})


@pytest.mark.parametrize(
    "bound", ["1/2", Decimal("0.5"), float("nan"), float("inf"), True]
)
@pytest.mark.parametrize("name", ["start", "end"])
def test_trim_fraction_unsupported_bounds(bound, name):
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[0, 1]],
        value=["a"],
    )
    with pytest.raises(BoppArgumentError, match="Trim bound must"):
        trim(ann, **{name: bound})


def test_trim_fraction_non_comparable_bounds():
    pt = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[0, 1]],
        value=["a"],
    )
    with pytest.raises(
        BoppArgumentError,
        match="Trim bound must be an int, finite float, Fraction, or pair of ints",
    ):
        trim(pt, start="1/2", end=Fraction(1))


@pytest.mark.parametrize("bound", [Decimal("NaN"), Decimal("sNaN"), "a"])
@pytest.mark.parametrize("name", ["start", "end"])
def test_trim_fraction_validates_bounds_before_order(bound, name):
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1]],
        duration=[[1, 1]],
        value=["a"],
    )
    bounds = {"start": 0, "end": 1, name: bound}
    with pytest.raises(
        BoppArgumentError,
        match="Trim bound must be an int, finite float, Fraction, or pair of ints",
    ):
        trim(ann, **bounds)


def test_trim_numeric_rejects_string_before_order():
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="time",
        time=[0.0, 1.0],
        value=list("ab"),
    )
    with pytest.raises(BoppArgumentError, match="Trim bound must be an int or float"):
        trim(ann, start="a", end=1)


@pytest.mark.parametrize(
    "extent_type", ["time", "time_interval", "midi_interval", None]
)
@pytest.mark.parametrize(
    "bound",
    [
        Decimal("0.5"),
        Decimal("NaN"),
        Decimal("sNaN"),
        math.nan,
        "a",
        True,
        np.float64(0.5),
        np.int64(1),
    ],
    ids=[
        "decimal",
        "decimal_nan",
        "decimal_snan",
        "nan",
        "string",
        "bool",
        "numpy_float",
        "numpy_int",
    ],
)
@pytest.mark.parametrize(
    "position", ["start", "end", "both", "start_with_end", "end_with_start"]
)
def test_trim_numeric_unsupported_bounds(extent_type, bound, position):
    if extent_type is None:
        ann = Annotation(
            media_id="track:1",
            bopp_version="1.0.0",
            metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
            extent=msgspec.UNSET,
            payload=TagPayload(value=["a"]),
        )
        bopp.ensure_annotation_id(ann)
    else:
        coords = {"tick": [0]} if extent_type == "midi_interval" else {"time": [0.0]}
        if extent_type != "time":
            coords["duration"] = [1]
        ann = create(
            media_id="track:1",
            payload_type="tag",
            extent_type=extent_type,
            value=["a"],
            **coords,
        )
    bounds = {
        "start": {"start": bound},
        "end": {"end": bound},
        "both": {"start": bound, "end": bound},
        "start_with_end": {"start": bound, "end": 2},
        "end_with_start": {"start": 0, "end": bound},
    }[position]
    message = (
        "Trim bound must not be NaN"
        if type(bound) is float
        else "Trim bound must be an int or float"
    )
    with pytest.raises(BoppArgumentError, match=message):
        trim(ann, **bounds)


def test_trim_no_extent_rejects_fraction_before_order():
    ann = Annotation(
        media_id="track:1",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=msgspec.UNSET,
        payload=TagPayload(value=["a"]),
    )
    bopp.ensure_annotation_id(ann)
    with pytest.raises(
        BoppArgumentError, match="Fraction or pair bounds require a fraction axis"
    ):
        trim(ann, start=[9, 1], end=5)


def test_trim_numeric_rejects_fraction_before_order():
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="time",
        time=[0.0, 1.0],
        value=list("ab"),
    )
    with pytest.raises(
        BoppArgumentError, match="Fraction or pair bounds require a fraction axis"
    ):
        trim(ann, start=[9, 1], end=5)


def test_trim_fraction_reversed_bounds_message():
    pt = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[0, 1]],
        value=["a"],
    )
    with pytest.raises(
        BoppArgumentError, match=r"start \(0\.3\) must be <= end \(0\.1\)"
    ):
        trim(pt, start=0.3, end=0.1)
    with pytest.raises(
        BoppArgumentError, match=r"start \(\[3, 1\]\) must be <= end \(1\)"
    ):
        trim(pt, start=[3, 1], end=1)
    with pytest.raises(
        BoppArgumentError, match=r"start \(\[3, 1\]\) must be <= end \(1\)"
    ):
        trim(pt, start=(3, 1), end=1)


@pytest.mark.parametrize(
    "start, end",
    [([3, 1], 1), (Fraction(3), [1, 1]), (3.0, (1, 1)), ((3, 1), Fraction(1))],
)
def test_trim_fraction_reversed_bounds(start, end):
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[0, 1]],
        value=["a"],
    )
    with pytest.raises(BoppArgumentError, match="start .* must be <= end"):
        trim(ann, start=start, end=end)


@pytest.mark.parametrize("bound", [Fraction(1, 2), [1, 2], (1, 2)])
@pytest.mark.parametrize("name", ["start", "end"])
def test_trim_numeric_axis_rejects_fraction_bounds(bound, name):
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="time",
        time=[0.0, 1.0],
        value=list("ab"),
    )
    with pytest.raises(BoppArgumentError, match="bounds require a fraction axis"):
        trim(ann, **{name: bound})


def test_trim_midi_tick():
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="midi_tick",
        tick=[0, 240, 480, 960],
        value=list("abcd"),
    )
    result = trim(ann, start=240, end=480, reset=True)
    assert result.extent.tick == [0, 240]
    assert result.payload.value == list("bc")


def test_trim_midi_interval():
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="midi_interval",
        tick=[0, 240, 480, 960],
        duration=[480, 240, 480, 240],
        value=list("abcd"),
    )
    result = trim(ann, start=240, end=720)
    assert result.extent.tick == [240, 240, 480]
    assert result.extent.duration == [240, 240, 240]
    assert result.payload.value == list("abc")


def test_trim_axis_configuration_keys():
    for tag, field, kind, span in [
        ("midi_tick", "tick", "point", None),
        ("midi_interval", "tick", "origin_span", "duration"),
        ("score_quarter", "quarter", "point", None),
        ("score_interval", "quarter", "origin_span", "duration"),
    ]:
        assert DEFAULT_TARGET_FIELDS[tag] == field
        assert AXIS_CONFIGS[(tag, field)] == (kind, field, span)
    assert FRACTION_AXES == frozenset(
        {
            ("score_quarter", "quarter"),
            ("score_interval", "quarter"),
        }
    )


def test_filter_by_fraction_extent():
    ann = create(
        media_id="track:1",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[-1, 2], [2, 4], [1, 1], [3, 2]],
        duration=[[1, 2], [1, 2], [2, 1], [1, 1]],
        value=list("abcd"),
    )
    result = filter_by(
        ann, lambda q: Fraction(*q) >= 1, facet="extent", target="quarter"
    )
    assert [list(q) for q in result.extent.quarter] == [[1, 1], [3, 2]]
    assert [list(d) for d in result.extent.duration] == [[2, 1], [1, 1]]
    assert result.payload.value == list("cd")


def test_filter_by_all_name_collision():
    ann = create(
        media_id="track:1",
        payload_type="beat",
        extent_type="time",
        time=[0.0, 1.0],
        value=[1, 1],
        groupings=[Grouping(name="time", ids=[0, 1], disjoint=False, exhaustive=False)],
    )
    with pytest.raises(
        BoppArgumentError,
        match="Ambiguous target field 'time' found in multiple facets when facet='all'",
    ):
        filter_by(ann, lambda x: x > 0, facet="all", target="time")

    with pytest.raises(
        BoppArgumentError,
        match=r"Column name collision across facets for \['time'\] when facet='all'",
    ):
        filter_by(ann, lambda r: r.time > 0, facet="all")


SCORE_CONVERSIONS = [
    (
        "to_score_quarter",
        "score_quarter_float",
        "score_quarter",
        {"max_denominator": 12},
    ),
    (
        "to_score_interval",
        "score_interval_float",
        "score_interval",
        {"max_denominator": 12},
    ),
    ("to_score_quarter_float", "score_quarter", "score_quarter_float", {}),
    ("to_score_interval_float", "score_interval", "score_interval_float", {}),
]


def _score_annotation(tag, **kwargs):
    """Create representative score columns, including zero and negative positions."""
    columns = (
        {"quarter": [-0.5, 0.0, 0.5, 1.0 / 3.0, 3.0]}
        if tag.endswith("_float")
        else {"quarter": [[-1, 2], [0, 1], [2, 4], [1, 3], [3, 1]]}
    )
    if "interval" in tag:
        columns["duration"] = (
            [0.0, 0.5, 1.0 / 3.0, 3.0, 0.125]
            if tag.endswith("_float")
            else [[0, 1], [2, 4], [1, 3], [3, 1], [1, 8]]
        )
    return create(
        media_id="track:score",
        payload_type="tag",
        extent_type=tag,
        value=list("abcde"),
        **columns,
        **kwargs,
    )


@pytest.mark.parametrize("name, source, target, kwargs", SCORE_CONVERSIONS)
@pytest.mark.parametrize("decoded", [False, True])
def test_score_conversion_values_and_provenance(name, source, target, kwargs, decoded):
    ann = _score_annotation(source)
    if decoded:
        ann = msgspec.json.decode(msgspec.json.encode(ann), type=type(ann))
    original = copy.deepcopy(ann)
    convert = getattr(bopp, name)
    assert convert is getattr(bopp.transforms, name)
    result = convert(ann, **kwargs)
    target_cls = bopp.registries.get_registry(ann.bopp_version)["EXTENT_TYPE_REGISTRY"][
        target
    ]
    assert type(result.extent) is target_cls
    if target.endswith("_float"):
        assert result.extent.quarter == [-0.5, 0.0, 0.5, 1.0 / 3.0, 3.0]
        if "interval" in target:
            assert result.extent.duration == [0.0, 0.5, 1.0 / 3.0, 3.0, 0.125]
        assert all(type(q) is float for q in result.extent.quarter)
    else:
        assert result.extent.quarter == [[-1, 2], [0, 1], [1, 2], [1, 3], [3, 1]]
        if "interval" in target:
            assert result.extent.duration == [[0, 1], [1, 2], [1, 3], [3, 1], [1, 8]]
        for field in msgspec.structs.fields(result.extent):
            val = getattr(result.extent, field.name)
            if isinstance(val, list):
                for pair in val:
                    assert type(pair) is list
                    assert pair[1] > 0
                    assert math.gcd(*pair) == 1
    assert result.parents == [ann.id]
    assert result.id != ann.id
    assert result.id == bopp.compute_annotation_id(result)
    assert result.media_id == ann.media_id
    assert result.bopp_version == ann.bopp_version
    assert result.payload == ann.payload
    assert isinstance(result.metadata, DerivedAnnotationMetadata)
    assert result.metadata.transform == name
    assert result.metadata.parameters == kwargs
    assert bopp.validate(result)
    assert ann == original
    loaded = msgspec.json.decode(msgspec.json.encode(result), type=type(result))
    assert loaded.id == result.id
    assert bopp.validate(loaded)
    assert msgspec.json.encode(loaded.extent) == msgspec.json.encode(result.extent)


@pytest.mark.parametrize("shape", ["quarter", "interval"])
@pytest.mark.parametrize(
    "value, bound, expected",
    [
        (0.3333333333333333, 1, [0, 1]),
        (0.3333333333333333, 2, [1, 2]),
        (0.3333333333333333, 3, [1, 3]),
        (2.2, 5, [11, 5]),
        (1.1, 5, [6, 5]),
        (-1.1, 5, [-6, 5]),
    ],
)
def test_score_conversion_denominator_bound(shape, value, bound, expected):
    ann = create(
        media_id="track:score",
        payload_type="tag",
        extent_type=f"score_{shape}_float",
        quarter=[value],
        value=["a"],
        **({"duration": [abs(value)]} if shape == "interval" else {}),
    )
    result = getattr(bopp, f"to_score_{shape}")(ann, max_denominator=bound)
    assert result.extent.quarter == [expected]
    if shape == "interval":
        assert result.extent.duration == [[abs(expected[0]), expected[1]]]


@pytest.mark.parametrize("shape", ["quarter", "interval"])
def test_score_conversion_roundtrips(shape):
    source = _score_annotation(f"score_{shape}")
    # Use reduced fractions to check exact recovery, including nondyadic values.
    source.extent.quarter = [[0, 1], [1, 3], [-3, 4], [11, 5], [1, 7]]
    if shape == "interval":
        source.extent.duration = [[0, 1], [1, 2], [1, 3], [3, 1], [1, 8]]
    source.id = bopp.compute_annotation_id(source)
    floating = getattr(bopp, f"to_score_{shape}_float")
    rational = getattr(bopp, f"to_score_{shape}")
    intermediate = floating(source)
    result = rational(intermediate, max_denominator=8)
    assert result.extent == source.extent
    assert result.parents == [source.id, intermediate.id]

    source = create(
        media_id="track:score",
        payload_type="tag",
        extent_type=f"score_{shape}_float",
        quarter=[0.0, 0.125, -0.75, 2.375],
        value=list("abcd"),
        **({"duration": [0.0, 0.125, 0.75, 2.375]} if shape == "interval" else {}),
    )
    intermediate = rational(source, max_denominator=8)
    result = floating(intermediate)
    assert result.extent == source.extent
    assert result.parents == [source.id, intermediate.id]


@pytest.mark.parametrize(
    "name, source, target, kwargs, tag",
    [
        (*conversion, tag)
        for conversion in SCORE_CONVERSIONS
        for tag in [
            "score_quarter",
            "score_interval",
            "score_quarter_float",
            "score_interval_float",
            "time",
            "midi_tick",
            None,
        ]
        if tag != conversion[1]
    ],
)
def test_score_conversion_rejects_wrong_source(name, source, target, kwargs, tag):
    if tag is not None and tag.startswith("score_"):
        ann = _score_annotation(tag)
    else:
        columns = (
            {"time": [0.0]}
            if tag == "time"
            else {"tick": [0]}
            if tag == "midi_tick"
            else {}
        )
        ann = create(
            media_id="track:score",
            payload_type="tag",
            extent_type=tag,
            value=["a"],
            **columns,
        )
    original = copy.deepcopy(ann)
    with pytest.raises(BoppArgumentError, match=rf"{name}:.*{source}"):
        getattr(bopp, name)(ann, **kwargs)
    assert ann == original
    if tag is None:
        ann.extent = None
        with pytest.raises(BoppArgumentError, match=rf"{name}:.*{source}"):
            getattr(bopp, name)(ann, **kwargs)


@pytest.mark.parametrize("shape", ["quarter", "interval"])
@pytest.mark.parametrize(
    "bound", [True, False, 2.0, 1.5, 0, -1, None, "8", np.int64(8)]
)
def test_score_conversion_invalid_denominator(shape, bound):
    ann = _score_annotation(f"score_{shape}_float")
    with pytest.raises(BoppArgumentError, match="max_denominator"):
        getattr(bopp, f"to_score_{shape}")(ann, max_denominator=bound)


@pytest.mark.parametrize("shape", ["quarter", "interval"])
def test_score_conversion_required_keyword(shape):
    ann = _score_annotation(f"score_{shape}_float")
    convert = getattr(bopp, f"to_score_{shape}")
    with pytest.raises(TypeError, match="max_denominator"):
        convert(ann)
    with pytest.raises(TypeError):
        convert(ann, 8)


@pytest.mark.parametrize(
    "shape, field",
    [
        ("quarter", "quarter"),
        ("interval", "quarter"),
        ("interval", "duration"),
    ],
)
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_score_conversion_nonfinite(shape, field, value):
    ann = _score_annotation(f"score_{shape}_float")
    getattr(ann.extent, field)[1] = value
    with pytest.raises(BoppArgumentError, match=rf"{field}.*index 1"):
        getattr(bopp, f"to_score_{shape}")(ann, max_denominator=8)


@pytest.mark.parametrize("name, source, target, kwargs", SCORE_CONVERSIONS)
def test_score_conversion_lineage_and_independent_facets(name, source, target, kwargs):
    ann = _score_annotation(
        source,
        confidence_type="likelihood",
        confidence=[0.9, 0.8, 0.7, 0.6, 0.5],
        sandbox={"custom": {"values": [1, 2]}},
    )
    trimmed = trim(ann, start=0)
    original = copy.deepcopy(trimmed)
    result = getattr(bopp, name)(trimmed, **kwargs)
    assert result.parents == [ann.id, trimmed.id]
    assert result.payload == trimmed.payload
    assert result.confidence == trimmed.confidence
    assert result.sandbox == trimmed.sandbox
    assert result.payload is not trimmed.payload
    assert result.confidence is not trimmed.confidence
    assert result.sandbox is not trimmed.sandbox
    result.payload.value.append("f")
    result.confidence.confidence.append(0.4)
    result.sandbox["custom"]["values"].append(3)
    result.parents.append("another")
    assert trimmed == original

    no_id = Annotation(
        media_id=ann.media_id,
        bopp_version=ann.bopp_version,
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=ann.extent,
        payload=ann.payload,
    )
    with pytest.warns(UserWarning, match="Converting an annotation with no ID"):
        result = getattr(bopp, name)(no_id, **kwargs)
    assert result.parents == []
    assert result.sandbox is msgspec.UNSET
    assert result.confidence is msgspec.UNSET
    assert bopp.validate(result)


@pytest.mark.parametrize("name, source, target, kwargs", SCORE_CONVERSIONS)
def test_score_conversion_empty_columns(name, source, target, kwargs):
    ann = create(
        media_id="track:score",
        payload_type="tag",
        extent_type=source,
        quarter=[],
        value=[],
        **({"duration": []} if "interval" in source else {}),
    )
    result = getattr(bopp, name)(ann, **kwargs)
    assert result.extent.quarter == []
    if "interval" in source:
        assert result.extent.duration == []
    assert bopp.validate(result)


@pytest.mark.parametrize("name, source, target, kwargs", SCORE_CONVERSIONS)
def test_score_conversion_preserves_groupings(name, source, target, kwargs):
    g = Grouping(name="layer", ids=[0, 1, 0, 1, 2], disjoint=False, exhaustive=False)
    ann = _score_annotation(source, groupings=[g])
    result = getattr(bopp, name)(ann, **kwargs)
    assert result.groupings is not None
    assert len(result.groupings) == 1
    assert result.groupings[0].name == "layer"
    assert result.groupings[0].ids == [0, 1, 0, 1, 2]
    assert result.groupings is not ann.groupings
    assert bopp.validate(result)


def test_marker_payload_transforms():
    ann = create(
        media_id="track:marker_test",
        payload_type="marker",
        extent_type="time",
        time=[0.1, 0.5, 1.0, 2.0],
        description="cues",
    )
    trimmed = trim(ann, start=0.4, end=1.5)
    assert trimmed.extent.time == [0.5, 1.0]
    assert trimmed.payload.description == "cues"

    filtered = filter_by(ann, lambda t: t > 0.5, facet="extent", target="time")
    assert filtered.extent.time == [1.0, 2.0]
    assert filtered.payload.description == "cues"

    resampled = to_times(ann, times=[0.0, 0.5, 1.5], method="nearest")
    assert resampled.extent.time == [0.0, 0.5, 1.5]
    assert resampled.payload.description == "cues"


def test_lyrics_payload_transforms():
    ann = create(
        media_id="track:lyrics_test",
        payload_type="lyrics",
        extent_type="time",
        time=[0.1, 0.5, 1.0, 2.0],
        value=["hello", "world", "sing", "song"],
        language="en-US",
    )
    trimmed = trim(ann, start=0.4, end=1.5)
    assert trimmed.extent.time == [0.5, 1.0]
    assert trimmed.payload.value == ["world", "sing"]
    assert trimmed.payload.language == "en-US"

    filtered = filter_by(ann, lambda v: len(v) == 4, facet="payload", target="value")
    assert filtered.payload.value == ["sing", "song"]
    assert filtered.payload.language == "en-US"


def test_frequency_hz_with_voiced_transforms():
    ann = create(
        media_id="track:freq_voiced_test",
        payload_type="frequency_hz",
        extent_type="time",
        time=[0.0, 0.5, 1.0, 1.5],
        value=[440.0, 442.0, 0.0, 880.0],
        voiced=[True, True, False, True],
    )
    trimmed = trim(ann, start=0.2, end=1.2)
    assert trimmed.extent.time == [0.5, 1.0]
    assert trimmed.payload.value == [442.0, 0.0]
    assert trimmed.payload.voiced == [True, False]

    # Filter by voiced flag using target
    voiced_only = filter_by(ann, lambda v: bool(v), target="voiced")
    assert voiced_only.extent.time == [0.0, 0.5, 1.5]
    assert voiced_only.payload.value == [440.0, 442.0, 880.0]
    assert voiced_only.payload.voiced == [True, True, True]

    # Filter using facet="all" multi-column record access
    voiced_all = filter_by(ann, lambda r: r.voiced, facet="all")
    assert voiced_all.payload.value == [440.0, 442.0, 880.0]


def test_object_with_description_transforms():
    ann = create(
        media_id="track:object_desc_test",
        payload_type="object",
        extent_type="time",
        time=[0.1, 0.5],
        value=[{"track": 1}, {"track": 2}],
        description="multi-track features",
    )
    trimmed = trim(ann, start=0.2)
    assert trimmed.extent.time == [0.5]
    assert trimmed.payload.value == [{"track": 2}]
    assert trimmed.payload.description == "multi-track features"


def test_note_payload_transforms():
    ann = create(
        media_id="track:note_spn_test",
        payload_type="note",
        extent_type="time",
        time=[0.0, 1.0, 2.0],
        value=["C4", "C♯4", "E♭3"],
    )
    trimmed = trim(ann, start=0.5, end=1.5)
    assert trimmed.extent.time == [1.0]
    assert trimmed.payload.value == ["C♯4"]


def test_trim_exhaustive_time_interval():
    ann = create(
        media_id="trim_ti_ex",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 10.0, 20.0],
        duration=[10.0, 10.0, 10.0],
        value=["a", "b", "c"],
        exhaustive=True,
        min_time=0.0,
        max_time=30.0,
    )
    # reset=True narrows and shifts bounds
    t_reset = trim(ann, start=5.0, end=25.0, reset=True)
    assert t_reset.extent.time == [0.0, 5.0, 15.0]
    assert t_reset.extent.duration == [5.0, 10.0, 5.0]
    assert t_reset.extent.min_time == 0.0
    assert t_reset.extent.max_time == 20.0
    assert t_reset.extent.exhaustive is True

    # reset=False narrows bounds without shifting
    t_no_reset = trim(ann, start=5.0, end=25.0, reset=False)
    assert t_no_reset.extent.time == [5.0, 10.0, 20.0]
    assert t_no_reset.extent.duration == [5.0, 10.0, 5.0]
    assert t_no_reset.extent.min_time == 5.0
    assert t_no_reset.extent.max_time == 25.0
    assert t_no_reset.extent.exhaustive is True

    # strict=True discards boundary-overlapping observations, sets exhaustive=False
    t_strict = trim(ann, start=5.0, end=25.0, strict=True)
    assert t_strict.extent.time == [10.0]
    assert t_strict.extent.duration == [10.0]
    assert t_strict.extent.min_time == 5.0
    assert t_strict.extent.max_time == 25.0
    assert t_strict.extent.exhaustive is False

    # Trimming with only min_time set
    ann_min_only = create(
        media_id="trim_min_only",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 10.0, 20.0],
        duration=[10.0, 10.0, 10.0],
        value=["a", "b", "c"],
        exhaustive=True,
        min_time=0.0,
    )
    t_min = trim(ann_min_only, start=5.0, reset=True)
    assert t_min.extent.min_time == 0.0
    assert t_min.extent.max_time is msgspec.UNSET

    # Trimming with only max_time set
    ann_max_only = create(
        media_id="trim_max_only",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 10.0, 20.0],
        duration=[10.0, 10.0, 10.0],
        value=["a", "b", "c"],
        exhaustive=True,
        max_time=30.0,
    )
    t_max = trim(ann_max_only, end=25.0, reset=False)
    assert t_max.extent.min_time is msgspec.UNSET
    assert t_max.extent.max_time == 25.0

    # Trimming out of bounds (empty intersection: coord_min > coord_max clamps)
    t_oob = trim(ann, start=35.0, end=40.0, reset=False)
    assert t_oob.extent.time == []
    assert t_oob.extent.min_time == 35.0
    assert t_oob.extent.max_time == 35.0
    assert t_oob.extent.exhaustive is False

    # Trimming without explicit bounds (min_time=UNSET, max_time=UNSET)
    ann_no_bounds = create(
        media_id="trim_no_bounds",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 10.0, 20.0],
        duration=[10.0, 10.0, 10.0],
        value=["a", "b", "c"],
        exhaustive=True,
    )
    t_nb = trim(ann_no_bounds, start=5.0, end=25.0, reset=True)
    assert t_nb.extent.min_time is msgspec.UNSET
    assert t_nb.extent.max_time is msgspec.UNSET
    assert t_nb.extent.exhaustive is True


def test_trim_exhaustive_time_uniform():
    ann = create(
        media_id="tu_ex",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        value=["a", "b", "c", "d", "e"],
        exhaustive=True,
        min_time=0.0,
        max_time=4.0,
        groupings=[{"name": "layer", "ids": [0, 0, 0, 0, 0], "exhaustive": True}],
    )
    # reset=True shifts start_time and bounds
    t_reset = trim(ann, start=1.0, end=3.0, reset=True)
    assert t_reset.extent.start_time == 0.0
    assert t_reset.extent.min_time == 0.0
    assert t_reset.extent.max_time == 2.0
    assert t_reset.extent.exhaustive is True
    assert t_reset.payload.value == ["b", "c", "d"]
    assert t_reset.groupings[0].exhaustive is True

    # reset=False preserves absolute coordinates
    t_no_reset = trim(ann, start=1.0, end=3.0, reset=False)
    assert t_no_reset.extent.start_time == 1.0
    assert t_no_reset.extent.min_time == 1.0
    assert t_no_reset.extent.max_time == 3.0
    assert t_no_reset.extent.exhaustive is True
    assert t_no_reset.groupings[0].exhaustive is True

    # strict=True forces exhaustive=False
    t_strict = trim(ann, start=1.0, end=3.0, strict=True)
    assert t_strict.extent.exhaustive is False
    assert t_strict.groupings[0].exhaustive is False

    # Gap beyond samples: start=0.2 < first sample (1.0) fails exhaustiveness check
    t_gap = trim(ann, start=0.2, end=3.0, reset=False)
    assert t_gap.extent.exhaustive is False

    # Trimming time_uniform with only min_time set
    ann_min_only = create(
        media_id="tu_min_only",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        value=["a", "b", "c", "d", "e"],
        exhaustive=True,
        min_time=0.0,
    )
    t_tu_min = trim(ann_min_only, start=1.0, reset=True)
    assert t_tu_min.extent.min_time == 0.0
    assert t_tu_min.extent.max_time is msgspec.UNSET

    # Trimming time_uniform with only max_time set
    ann_max_only = create(
        media_id="tu_max_only",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        value=["a", "b", "c", "d", "e"],
        exhaustive=True,
        max_time=4.0,
    )
    t_tu_max = trim(ann_max_only, end=3.0, reset=False)
    assert t_tu_max.extent.min_time is msgspec.UNSET
    assert t_tu_max.extent.max_time == 3.0

    # Trimming time_uniform out of bounds (coord_min > coord_max clamps)
    t_tu_oob = trim(ann, start=10.0, end=20.0, reset=False)
    assert t_tu_oob.payload.value == []
    assert t_tu_oob.extent.min_time == 10.0
    assert t_tu_oob.extent.max_time == 10.0
    assert t_tu_oob.extent.exhaustive is False


def test_trim_exhaustive_with_groupings():
    ann = create(
        media_id="g_ex",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 10.0, 20.0],
        duration=[10.0, 10.0, 10.0],
        value=["a", "b", "c"],
        exhaustive=True,
        min_time=0.0,
        max_time=30.0,
        groupings=[
            {"name": "full", "ids": [0, 0, 0], "exhaustive": True},
            {"name": "partial", "ids": [-1, 0, 0], "exhaustive": False},
        ],
    )
    trimmed = trim(ann, start=5.0, end=25.0, reset=True)
    assert trimmed.groupings[0].exhaustive is True
    assert trimmed.groupings[1].exhaustive is False

    trimmed_strict = trim(ann, start=5.0, end=25.0, strict=True)
    assert trimmed_strict.groupings[0].exhaustive is False
    assert trimmed_strict.groupings[1].exhaustive is False


def test_trim_exhaustive_fraction_and_integer_axes():
    # Fraction axis (score_interval)
    ann_score = create(
        media_id="score_ex",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1], [1, 1], [2, 1]],
        duration=[[1, 1], [1, 1], [1, 1]],
        value=["a", "b", "c"],
        exhaustive=True,
        min_quarter=[0, 1],
        max_quarter=[3, 1],
    )
    t_score_reset = trim(ann_score, start=[1, 2], end=[5, 2], reset=True)
    assert t_score_reset.extent.min_quarter == [0, 1]
    assert t_score_reset.extent.max_quarter == [2, 1]
    assert t_score_reset.extent.exhaustive is True

    t_score_no_reset = trim(ann_score, start=[1, 2], end=[5, 2], reset=False)
    assert t_score_no_reset.extent.min_quarter == [1, 2]
    assert t_score_no_reset.extent.max_quarter == [5, 2]
    assert t_score_no_reset.extent.exhaustive is True

    # Integer axis (text_interval)
    ann_text = create(
        media_id="text_ex",
        payload_type="tag",
        extent_type="text_interval",
        offset=[0, 5, 10],
        length=[5, 5, 5],
        value=["a", "b", "c"],
        exhaustive=True,
        min_offset=0,
        max_offset=15,
    )
    t_text_reset = trim(ann_text, start=2, end=12, reset=True)
    assert t_text_reset.extent.min_offset == 0
    assert t_text_reset.extent.max_offset == 10
    assert t_text_reset.extent.offset == [0, 3, 8]
    assert t_text_reset.extent.length == [3, 5, 2]
    assert t_text_reset.extent.exhaustive is True

    t_text_no_reset = trim(ann_text, start=2, end=12, reset=False)
    assert t_text_no_reset.extent.min_offset == 2
    assert t_text_no_reset.extent.max_offset == 12
    assert t_text_no_reset.extent.offset == [2, 5, 10]
    assert t_text_no_reset.extent.length == [3, 5, 2]
    assert t_text_no_reset.extent.exhaustive is True


def test_filter_by_exhaustive_extent_and_groupings():
    ann = create(
        media_id="filter_ex",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 10.0, 20.0],
        duration=[10.0, 10.0, 10.0],
        value=["keep1", "drop", "keep2"],
        exhaustive=True,
        min_time=0.0,
        max_time=30.0,
        groupings=[{"name": "layer", "ids": [0, 0, 0], "exhaustive": True}],
    )
    # Dropping observations sets exhaustive=False
    filtered_dropped = filter_by(ann, lambda v: v != "drop")
    assert filtered_dropped.extent.exhaustive is False
    assert filtered_dropped.groupings[0].exhaustive is False
    assert filtered_dropped.extent.min_time == 0.0
    assert filtered_dropped.extent.max_time == 30.0

    # Keeping all observations preserves exhaustive=True
    filtered_all = filter_by(ann, lambda v: True)
    assert filtered_all.extent.exhaustive is True
    assert filtered_all.groupings[0].exhaustive is True


def test_format_bound_value_and_metadata_helpers():
    from bopp.transforms import _format_bound_value, _get_extent_axis_bound_attrs

    # Fraction formatting
    assert _format_bound_value([1, 2], fraction_axis=True, integer_axis=False) == [1, 2]
    assert _format_bound_value(
        Fraction(3, 4), fraction_axis=True, integer_axis=False
    ) == [3, 4]
    assert _format_bound_value(2, fraction_axis=True, integer_axis=False) == [2, 1]

    # Integer formatting
    assert _format_bound_value(5.0, fraction_axis=False, integer_axis=True) == 5
    assert _format_bound_value(7, fraction_axis=False, integer_axis=True) == 7

    # Float formatting
    assert _format_bound_value(3, fraction_axis=False, integer_axis=False) == 3.0
    assert _format_bound_value(4.5, fraction_axis=False, integer_axis=False) == 4.5

    # Unsupported axis metadata raises BoppArgumentError
    with pytest.raises(BoppArgumentError):
        _get_extent_axis_bound_attrs("time", "nonexistent_field")
    with pytest.raises(BoppArgumentError):
        _get_extent_axis_bound_attrs("nonexistent_tag", "time")


def test_to_times_resamples_groupings_exhaustive_false():
    from bopp.transforms import to_times

    ann = create(
        media_id="to_times_ex",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 10.0],
        duration=[10.0, 10.0],
        value=["a", "b"],
        groupings=[{"name": "layer", "ids": [0, 0], "exhaustive": True}],
    )
    res = to_times(ann, times=[0.0, 5.0, 15.0])
    assert res.extent.exhaustive is False
    assert res.groupings[0].exhaustive is False


# ==============================================================================
# Tests for sort
# ==============================================================================


def test_sort_payload_ascending_and_descending():
    ann = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time",
        time=[3.0, 1.0, 2.0],
        value=["c", "a", "b"],
    )
    res_asc = sort(ann, by="value")
    assert res_asc.payload.value == ["a", "b", "c"]
    assert res_asc.extent.time == [1.0, 2.0, 3.0]
    assert res_asc.metadata.transform == "sort"
    assert res_asc.metadata.parameters == {"by": ["value"], "ascending": True}
    assert res_asc.parents == [ann.id]

    res_desc = sort(ann, by="value", ascending=False)
    assert res_desc.payload.value == ["c", "b", "a"]
    assert res_desc.extent.time == [3.0, 2.0, 1.0]
    assert res_desc.metadata.parameters == {"by": ["value"], "ascending": False}


def test_sort_qualified_facets():
    ann = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        time=[10.0, 30.0, 20.0],
        value=["x", "y", "z"],
        confidence=[0.5, 0.1, 0.9],
    )
    res_extent = sort(ann, by="extent.time")
    assert res_extent.extent.time == [10.0, 20.0, 30.0]
    assert res_extent.confidence.confidence == [0.5, 0.9, 0.1]

    res_conf = sort(ann, by="confidence.confidence")
    assert res_conf.confidence.confidence == [0.1, 0.5, 0.9]
    assert res_conf.extent.time == [30.0, 10.0, 20.0]

    res_payload = sort(ann, by="payload.value", ascending=False)
    assert res_payload.payload.value == ["z", "y", "x"]


def test_sort_groupings():
    ann_single = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time",
        time=[3.0, 1.0, 2.0],
        value=["c", "a", "b"],
        groupings=[{"name": "section", "ids": [2, 0, 1]}],
    )
    res_single = sort(ann_single, by="grouping")
    assert res_single.groupings[0].ids == [0, 1, 2]
    assert res_single.payload.value == ["a", "b", "c"]

    res_single_plural = sort(ann_single, by="groupings")
    assert res_single_plural.groupings[0].ids == [0, 1, 2]

    ann_multi = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time",
        time=[4.0, 3.0, 2.0, 1.0],
        value=["d", "c", "b", "a"],
        groupings=[
            {"name": "layer_a", "ids": [1, 0, 1, 0]},
            {"name": "layer_b", "ids": [3, 2, 1, 0]},
        ],
    )
    # Ambiguous grouping error
    with pytest.raises(
        BoppArgumentError, match="ambiguous because annotation has 2 groupings"
    ):
        sort(ann_multi, by="grouping")

    # Index notation
    res_idx0 = sort(ann_multi, by="grouping.0")
    assert res_idx0.groupings[0].ids == [0, 0, 1, 1]

    res_group_dot = sort(ann_multi, by="group.1")
    assert res_group_dot.groupings[1].ids == [0, 1, 2, 3]

    # Name notation
    res_name = sort(ann_multi, by="grouping.layer_b")
    assert res_name.groupings[1].ids == [0, 1, 2, 3]

    # Bare grouping name
    res_bare = sort(ann_multi, by="layer_a")
    assert res_bare.groupings[0].ids == [0, 0, 1, 1]


def test_sort_tie_breaking_multiple_keys():
    ann = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time",
        time=[3.0, 1.0, 2.0, 4.0],
        value=["b", "a", "c", "d"],
        groupings=[{"name": "sec", "ids": [1, 1, 0, 0]}],
    )
    # Sort first by grouping ascending, then by time descending
    res = sort(ann, by=["sec", "extent.time"], ascending=[True, False])
    assert res.groupings[0].ids == [0, 0, 1, 1]
    assert res.extent.time == [4.0, 2.0, 3.0, 1.0]
    assert res.payload.value == ["d", "c", "b", "a"]


def test_sort_fractions():
    ann = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[1, 2], [1, 4], [3, 8]],
        value=["half", "fourth", "three-eighths"],
    )
    res = sort(ann, by="quarter")
    assert res.extent.quarter == [[1, 4], [3, 8], [1, 2]]
    assert res.payload.value == ["fourth", "three-eighths", "half"]

    res_desc = sort(ann, by="extent.quarter", ascending=False)
    assert res_desc.extent.quarter == [[1, 2], [3, 8], [1, 4]]


def test_sort_time_uniform_raises():
    ann = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=0.5,
        value=["a", "b", "c"],
    )
    with pytest.raises(
        BoppArgumentError, match="sort is not supported for 'time_uniform' extents"
    ):
        sort(ann, by="value")


def test_sort_validation_errors():
    ann = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0],
        value=["a", "b"],
    )
    with pytest.raises(BoppArgumentError, match="'by' must not be an empty sequence"):
        sort(ann, by=[])

    with pytest.raises(BoppArgumentError, match="must be a str or Sequence"):
        sort(ann, by=123)  # type: ignore[arg-type]

    with pytest.raises(BoppArgumentError, match="Sort key must be a string"):
        sort(ann, by=[123])  # type: ignore[list-item]

    with pytest.raises(BoppArgumentError, match="Length of 'ascending'"):
        sort(ann, by=["value", "time"], ascending=[True])

    with pytest.raises(
        BoppArgumentError, match="'ascending' elements must be booleans"
    ):
        sort(ann, by=["value"], ascending=["yes"])  # type: ignore[list-item]

    with pytest.raises(
        BoppArgumentError, match="'ascending' must be a bool or Sequence"
    ):
        sort(ann, by="value", ascending="true")  # type: ignore[arg-type]

    with pytest.raises(BoppArgumentError, match="Sort key 'unknown' not found"):
        sort(ann, by="unknown")

    with pytest.raises(
        BoppArgumentError, match="Field 'missing' not found in payload columns"
    ):
        sort(ann, by="payload.missing")

    with pytest.raises(
        BoppArgumentError, match="Field 'missing' not found in extent columns"
    ):
        sort(ann, by="extent.missing")

    with pytest.raises(BoppArgumentError, match="Unrecognized facet prefix"):
        sort(ann, by="invalid_facet.value")

    with pytest.raises(BoppArgumentError, match="Grouping index 5 out of range"):
        ann_with_g = create(
            media_id="track_1",
            payload_type="tag",
            extent_type="time",
            time=[1.0],
            value=["a"],
            groupings=[{"name": "g1", "ids": [0]}],
        )
        sort(ann_with_g, by="grouping.5")

    with pytest.raises(BoppArgumentError, match="Grouping layer 'missing' not found"):
        sort(ann_with_g, by="grouping.missing")

    with pytest.raises(BoppArgumentError, match="annotation has no groupings"):
        sort(ann, by="grouping")


def test_sort_empty_and_single_observation():
    ann_empty = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time",
        time=[],
        value=[],
    )
    res_empty = sort(ann_empty, by="value")
    assert res_empty.payload.value == []
    assert res_empty.metadata.transform == "sort"

    ann_single = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["solo"],
    )
    res_single = sort(ann_single, by="value")
    assert res_single.payload.value == ["solo"]


# ==============================================================================
# Tests for shift
# ==============================================================================


def test_shift_1d_time():
    ann = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.5, 4.0],
        value=["a", "b", "c"],
        min_time=0.0,
        max_time=10.0,
    )
    shifted = shift(ann, 2.0)
    assert shifted.extent.time == [3.0, 4.5, 6.0]
    assert shifted.extent.min_time == 2.0
    assert shifted.extent.max_time == 12.0
    assert shifted.metadata.transform == "shift"
    assert shifted.metadata.parameters == {"displacement": 2.0, "target_field": "time"}

    shifted_neg = shift(ann, -1.0)
    assert shifted_neg.extent.time == [0.0, 1.5, 3.0]


def test_shift_time_interval():
    ann = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 3.0],
        duration=[2.0, 4.0],
        value=["verse", "chorus"],
    )
    shifted = shift(ann, 1.5)
    assert shifted.extent.time == [1.5, 4.5]
    assert shifted.extent.duration == [2.0, 4.0]  # Duration remains invariant


def test_shift_time_uniform():
    ann = create(
        media_id="track_1",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=1.0,
        sampling_period=0.25,
        min_time=1.0,
        max_time=5.0,
        value=["a", "b", "c"],
    )
    shifted = shift(ann, 0.5)
    assert shifted.extent.start_time == 1.5
    assert shifted.extent.sampling_period == 0.25
    assert shifted.extent.min_time == 1.5
    assert shifted.extent.max_time == 5.5

    with pytest.raises(BoppArgumentError, match="Unsupported target_field 'x'"):
        shift(ann, 0.5, target_field="x")


def test_shift_fraction_extent():
    ann = create(
        media_id="score_1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[0, 1], [1, 2]],
        value=["a", "b"],
        min_quarter=[0, 1],
        max_quarter=[2, 1],
    )
    shifted_frac = shift(ann, Fraction(1, 4))
    assert shifted_frac.extent.quarter == [[1, 4], [3, 4]]
    assert shifted_frac.extent.min_quarter == [1, 4]
    assert shifted_frac.extent.max_quarter == [9, 4]

    shifted_pair = shift(ann, [1, 2])
    assert shifted_pair.extent.quarter == [[1, 2], [1, 1]]

    shifted_int = shift(ann, 1)
    assert shifted_int.extent.quarter == [[1, 1], [3, 2]]

    # score_interval: duration remains invariant
    ann_int = create(
        media_id="score_1",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[1, 4]],
        duration=[[1, 2]],
        value=["c"],
    )
    shifted_interval = shift(ann_int, [1, 4])
    assert shifted_interval.extent.quarter == [[1, 2]]
    assert shifted_interval.extent.duration == [[1, 2]]


def test_shift_integer_extent():
    ann = create(
        media_id="text_1",
        payload_type="tag",
        extent_type="text_interval",
        offset=[5, 15],
        length=[3, 4],
        value=["word1", "word2"],
        min_offset=0,
        max_offset=50,
    )
    shifted = shift(ann, 10)
    assert shifted.extent.offset == [15, 25]
    assert shifted.extent.length == [3, 4]
    assert shifted.extent.min_offset == 10
    assert shifted.extent.max_offset == 60

    with pytest.raises(BoppArgumentError, match="must be an int"):
        shift(ann, 1.5)

    with pytest.raises(BoppArgumentError, match="must be an int"):
        shift(ann, True)  # bool rejected


def test_shift_multi_axis():
    ann_box = create(
        media_id="img_1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[10.0],
        width=[20.0],
        y=[30.0],
        height=[40.0],
        value=["car"],
        min_x=0.0,
        max_x=100.0,
        min_y=0.0,
        max_y=200.0,
    )
    # Must specify target_field or dict
    with pytest.raises(BoppArgumentError, match="does not have a default target field"):
        shift(ann_box, 5.0)

    # Shift single axis
    shifted_x = shift(ann_box, 5.0, target_field="x")
    assert shifted_x.extent.x == [15.0]
    assert shifted_x.extent.width == [20.0]
    assert shifted_x.extent.min_x == 5.0
    assert shifted_x.extent.y == [30.0]

    # Shift multiple axes via dictionary
    shifted_dict = shift(ann_box, {"x": 5.0, "y": -10.0})
    assert shifted_dict.extent.x == [15.0]
    assert shifted_dict.extent.y == [20.0]
    assert shifted_dict.extent.min_x == 5.0
    assert shifted_dict.extent.min_y == -10.0
    assert shifted_dict.extent.max_x == 105.0
    assert shifted_dict.extent.max_y == 190.0


def test_shift_validation_errors():
    ann_no_extent = create(
        media_id="global_1",
        payload_type="tag",
        value=["scene"],
    )
    with pytest.raises(
        BoppArgumentError, match="shift requires an annotation with an extent"
    ):
        shift(ann_no_extent, 1.0)

    ann_time = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
    )
    with pytest.raises(
        BoppArgumentError, match="Cannot specify both 'target_field' and a dictionary"
    ):
        shift(ann_time, {"time": 1.0}, target_field="time")

    with pytest.raises(
        BoppArgumentError, match="Dictionary 'displacement' must not be empty"
    ):
        shift(ann_time, {})

    with pytest.raises(BoppArgumentError, match="must be finite and not NaN"):
        shift(ann_time, float("nan"))

    with pytest.raises(BoppArgumentError, match="must be finite and not NaN"):
        shift(ann_time, float("inf"))

    with pytest.raises(
        BoppArgumentError, match="Unsupported target_field 'nonexistent'"
    ):
        shift(ann_time, 1.0, target_field="nonexistent")


# ==============================================================================
# Tests for union
# ==============================================================================


def test_union_two_compatible_annotations():
    ann1 = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=["verse", "chorus"],
        sandbox={"custom": "keep"},
    )
    ann2 = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time_interval",
        time=[4.0],
        duration=[3.0],
        value=["outro"],
        sandbox={"custom": "ignored"},
    )
    res = union(ann1, ann2)
    assert res.extent.time == [0.0, 2.0, 4.0]
    assert res.extent.duration == [2.0, 2.0, 3.0]
    assert res.payload.value == ["verse", "chorus", "outro"]
    assert res.sandbox == {"custom": "keep"}  # First operand propagated
    assert res.parents == [ann1.id, ann2.id]


def test_union_sequence_input_and_three_annotations():
    ann1 = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
    )
    ann2 = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["b"],
    )
    ann3 = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time",
        time=[3.0],
        value=["c"],
    )
    # As list
    res_list = union([ann1, ann2, ann3])
    assert res_list.extent.time == [1.0, 2.0, 3.0]
    assert res_list.parents == [ann1.id, ann2.id, ann3.id]

    # As *args
    res_args = union(ann1, ann2, ann3)
    assert res_args.extent.time == [1.0, 2.0, 3.0]


def test_union_groupings_and_labels():
    ann1 = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
        groupings=[{"name": "sec", "ids": [0], "labels": {"verse": 0}}],
    )
    ann2 = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["b"],
        groupings=[{"name": "sec", "ids": [1], "labels": {"chorus": 1}}],
    )
    res = union(ann1, ann2)
    assert res.groupings[0].ids == [0, 1]
    assert res.groupings[0].labels == {"verse": 0, "chorus": 1}

    # Conflict in labels
    ann_conflict = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time",
        time=[3.0],
        value=["c"],
        groupings=[{"name": "sec", "ids": [1], "labels": {"bridge": 1}}],
    )
    with pytest.raises(BoppArgumentError, match="Conflicting label mapping"):
        union(ann2, ann_conflict)


def test_union_recomputes_disjoint_and_exhaustive():
    # Disjoint intervals become non-disjoint
    ann1 = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[3.0],
        value=["a"],
    )
    ann2 = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time_interval",
        time=[2.0],
        duration=[3.0],
        value=["b"],
    )
    res = union(ann1, ann2)
    assert res.extent.disjoint is False


def test_union_validation_errors():
    ann1 = create(
        media_id="media_1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
    )
    with pytest.raises(BoppArgumentError, match="requires at least 2 annotations"):
        union(ann1)

    with pytest.raises(BoppArgumentError, match="is not an Annotation instance"):
        union(ann1, "not_an_annotation")  # type: ignore[arg-type]

    # Mismatched media_id
    ann_diff_media = create(
        media_id="media_2",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["b"],
    )
    with pytest.raises(BoppArgumentError, match="media_id"):
        union(ann1, ann_diff_media)

    # Mismatched payload_type
    ann_diff_payload = create(
        media_id="media_1",
        payload_type="chord",
        extent_type="time",
        time=[2.0],
        value=["C:maj"],
    )
    with pytest.raises(BoppArgumentError, match="Payload type mismatch"):
        union(ann1, ann_diff_payload)

    # Mismatched extent_type
    ann_diff_extent = create(
        media_id="media_1",
        payload_type="tag",
        extent_type="pixel",
        x=[1.0],
        y=[2.0],
        value=["a"],
    )
    with pytest.raises(BoppArgumentError, match="Extent type mismatch"):
        union(ann1, ann_diff_extent)

    # time_uniform extent error
    ann_tu = create(
        media_id="media_1",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        value=["a"],
    )
    with pytest.raises(
        BoppArgumentError, match="union does not support 'time_uniform' extents"
    ):
        union(ann_tu, ann_tu)


# ==============================================================================
# Tests for to_uniform_times
# ==============================================================================


def test_to_uniform_times_disjoint_intervals():
    ann = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=["verse", "chorus"],
    )
    res = to_uniform_times(ann, sampling_period=1.0)
    assert res.extent.start_time == 0.0
    assert res.extent.sampling_period == 1.0
    assert res.payload.value == ["verse", "verse", "chorus", "chorus", "chorus"]
    assert res.extent.disjoint is True
    assert res.metadata.transform == "to_uniform_times"


def test_to_uniform_times_disjoint_points():
    ann = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time",
        time=[0.0, 2.0],
        value=["verse", "chorus"],
    )
    res_prev = to_uniform_times(ann, sampling_period=0.5, method="previous")
    assert res_prev.payload.value == ["verse", "verse", "verse", "verse", "chorus"]

    res_near = to_uniform_times(ann, sampling_period=0.5, method="nearest")
    assert res_near.payload.value == [
        "verse",
        "verse",
        "verse",
        "chorus",
        "chorus",
    ]


def test_to_uniform_times_requires_disjoint_extent():
    # Overlapping intervals must raise error
    ann_overlap = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 1.0],
        duration=[2.0, 2.0],
        value=["a", "b"],
    )
    with pytest.raises(
        BoppArgumentError, match="requires the input extent to be disjoint"
    ):
        to_uniform_times(ann_overlap, sampling_period=1.0)

    # Grouping is disjoint, but the overall extent is not -> MUST STILL RAISE!
    ann_grouped_disjoint = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 1.0],
        duration=[2.0, 2.0],
        value=["a", "b"],
        groupings=[{"name": "track", "ids": [0, 1], "disjoint": True}],
    )
    with pytest.raises(
        BoppArgumentError, match="requires the input extent to be disjoint"
    ):
        to_uniform_times(ann_grouped_disjoint, sampling_period=1.0)


def test_to_uniform_times_gap_and_fill_value():
    ann = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.0],
        duration=[1.0],
        value=["solo"],
    )
    res = to_uniform_times(
        ann,
        sampling_period=0.5,
        start_time=0.0,
        end_time=3.0,
        fill_value="silence",
    )
    assert res.payload.value == [
        "silence",
        "silence",
        "solo",
        "solo",
        "solo",
        "silence",
        "silence",
    ]


def test_to_uniform_times_validation_errors():
    ann = create(
        media_id="song_1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[1.0],
        value=["a"],
    )
    with pytest.raises(BoppArgumentError, match="sampling_period must be positive"):
        to_uniform_times(ann, sampling_period=0)

    with pytest.raises(
        BoppArgumentError, match="sampling_period must be a positive number"
    ):
        to_uniform_times(ann, sampling_period=True)  # type: ignore[arg-type]

    with pytest.raises(BoppArgumentError, match="Invalid method"):
        to_uniform_times(ann, sampling_period=1.0, method="invalid")  # type: ignore[arg-type]

    with pytest.raises(BoppArgumentError, match="Invalid overlap strategy"):
        to_uniform_times(ann, sampling_period=1.0, overlap="multiple")  # type: ignore[arg-type]

    with pytest.raises(BoppArgumentError, match="start_time .* must be <= end_time"):
        to_uniform_times(ann, sampling_period=1.0, start_time=5.0, end_time=2.0)


# ==============================================================================
# Tests for to_pixels
# ==============================================================================


def test_to_pixels_pixel_box():
    ann = create(
        media_id="frame_1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0, 100.0],
        width=[50.0, 50.0],
        y=[0.0, 100.0],
        height=[50.0, 50.0],
        value=["top_left", "bottom_right"],
    )
    # Using pixels tuples
    res = to_pixels(ann, pixels=[(25.0, 25.0), (120.0, 120.0)])
    assert res.extent.x == [25.0, 120.0]
    assert res.extent.y == [25.0, 120.0]
    assert res.payload.value == ["top_left", "bottom_right"]
    assert res.metadata.transform == "to_pixels"

    # Using x and y sequences
    res_xy = to_pixels(ann, x=[25.0, 120.0], y=[25.0, 120.0])
    assert res_xy.payload.value == ["top_left", "bottom_right"]


def test_to_pixels_overlap_modes():
    ann = create(
        media_id="frame_1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0, 10.0],
        width=[20.0, 20.0],
        y=[0.0, 10.0],
        height=[20.0, 20.0],
        value=["box1", "box2"],
    )
    # Point (15, 15) is covered by both box1 [0..20, 0..20] and box2 [10..30, 10..30]
    res_latest = to_pixels(ann, pixels=[(15.0, 15.0)], overlap="latest")
    assert res_latest.payload.value == ["box2"]

    res_first = to_pixels(ann, pixels=[(15.0, 15.0)], overlap="first")
    assert res_first.payload.value == ["box1"]

    res_multiple = to_pixels(ann, pixels=[(15.0, 15.0)], overlap="multiple")
    assert res_multiple.payload.value == ["box1", "box2"]
    assert res_multiple.extent.x == [15.0, 15.0]


def test_to_pixels_point_pixel():
    ann = create(
        media_id="frame_1",
        payload_type="tag",
        extent_type="pixel",
        x=[10.0, 20.0],
        y=[30.0, 40.0],
        value=["p1", "p2"],
    )
    res = to_pixels(ann, pixels=[(10.0, 30.0), (99.0, 99.0)], fill_value="miss")
    assert res.payload.value == ["p1", "miss"]


def test_to_pixels_validation_errors():
    ann = create(
        media_id="frame_1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0],
        width=[10.0],
        y=[0.0],
        height=[10.0],
        value=["a"],
    )
    with pytest.raises(BoppArgumentError, match="Cannot specify both 'pixels' and"):
        to_pixels(ann, pixels=[(1.0, 1.0)], x=[1.0])

    with pytest.raises(
        BoppArgumentError, match="Lengths of 'x' .* and 'y' .* must match"
    ):
        to_pixels(ann, x=[1.0, 2.0], y=[1.0])

    with pytest.raises(BoppArgumentError, match="Must specify either 'pixels'"):
        to_pixels(ann)

    with pytest.raises(BoppArgumentError, match="Invalid overlap strategy"):
        to_pixels(ann, pixels=[(1.0, 1.0)], overlap="invalid")  # type: ignore[arg-type]

    # Incompatible extent
    ann_time = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
    )
    with pytest.raises(BoppArgumentError, match="is incompatible with to_pixels"):
        to_pixels(ann_time, pixels=[(1.0, 1.0)])


# ==============================================================================
# Additional Coverage Tests for sort, shift, union, to_uniform_times, to_pixels
# ==============================================================================


def test_sort_coverage_branches():
    # 1. Bare non-fraction extent column
    ann_bare_time = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        time=[3.0, 1.0, 2.0],
        value=["c", "a", "b"],
    )
    res_bare_time = sort(ann_bare_time, by="time")
    assert res_bare_time.extent.time == [1.0, 2.0, 3.0]

    # 2. Bare confidence column
    ann_bare_conf = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        time=[1.0, 2.0, 3.0],
        value=["a", "b", "c"],
        confidence=[0.9, 0.1, 0.5],
    )
    res_bare_conf = sort(ann_bare_conf, by="confidence")
    assert res_bare_conf.confidence.confidence == [0.1, 0.5, 0.9]

    # 3. Missing confidence column in qualified search
    with pytest.raises(
        BoppArgumentError, match="Field 'missing' not found in confidence"
    ):
        sort(ann_bare_conf, by="confidence.missing")

    # 4. Ambiguous sort key across facets
    # TagPayload has 'value'; create an extent or confidence that also has 'value'
    # Or create an annotation with Grouping named 'value'
    ann_ambig = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0],
        value=["a", "b"],
        groupings=[{"name": "value", "ids": [1, 0]}],
    )
    with pytest.raises(BoppArgumentError, match="Ambiguous sort key 'value'"):
        sort(ann_ambig, by="value")

    # 5. Bare fraction extent axis sort
    ann_frac = create(
        media_id="t1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[3, 1], [1, 1], [2, 1]],
        value=["c", "a", "b"],
    )
    res_frac = sort(ann_frac, by="quarter")
    assert res_frac.extent.quarter == [[1, 1], [2, 1], [3, 1]]

    # 6. Unrecognized facet prefix
    with pytest.raises(BoppArgumentError, match="Unrecognized facet prefix 'invalid'"):
        sort(ann_bare_time, by="invalid.field")


def test_shift_coverage_branches(monkeypatch):
    # 1. time_frequency_box on 'frequency' axis (kind == 'min_max')
    ann_tfbox = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time_frequency_box",
        time=[0.0],
        duration=[1.0],
        freq_min=[100.0],
        freq_max=[500.0],
        value=["tone"],
        min_freq=50.0,
        max_freq=1000.0,
    )
    shifted_freq = shift(ann_tfbox, 50.0, target_field="frequency")
    assert shifted_freq.extent.freq_min == [150.0]
    assert shifted_freq.extent.freq_max == [550.0]
    assert shifted_freq.extent.min_freq == 100.0
    assert shifted_freq.extent.max_freq == 1050.0

    # 2. text extent (integer axis, kind == 'point')
    ann_text = create(
        media_id="text_1",
        payload_type="tag",
        extent_type="text",
        offset=[5, 15],
        value=["w1", "w2"],
        min_offset=0,
        max_offset=20,
    )
    shifted_text = shift(ann_text, 10)
    assert shifted_text.extent.offset == [15, 25]
    assert shifted_text.extent.min_offset == 10
    assert shifted_text.extent.max_offset == 30

    # 3. time_uniform with no min_time / max_time
    ann_tu_nobounds = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=1.0,
        sampling_period=0.5,
        value=["a", "b"],
    )
    shifted_tu = shift(ann_tu_nobounds, 2.0)
    assert shifted_tu.extent.start_time == 3.0

    # 4. time_uniform with invalid displacement
    with pytest.raises(
        BoppArgumentError,
        match="Displacement for 'time_uniform' must be an int or float",
    ):
        shift(ann_tu_nobounds, "bad")  # type: ignore[arg-type]

    with pytest.raises(BoppArgumentError, match="must be finite and not NaN"):
        shift(ann_tu_nobounds, float("nan"))

    # 5. Fraction axis with invalid displacement
    ann_score = create(
        media_id="score_1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[0, 1]],
        value=["note"],
    )
    with pytest.raises(
        BoppArgumentError,
        match="Displacement for fraction axis 'quarter' must be an int",
    ):
        shift(ann_score, "invalid")  # type: ignore[arg-type]

    # 6. Extent without valid schema tag
    with monkeypatch.context() as m:
        m.setattr("bopp.transforms._get_tag", lambda _: None)
        with pytest.raises(BoppArgumentError, match="extent has no valid schema tag"):
            shift(ann_text, 1)

    # 7. Float axis with invalid type
    ann_time = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
    )
    with pytest.raises(
        BoppArgumentError, match="Displacement for axis 'time' must be an int or float"
    ):
        shift(ann_time, "bad")  # type: ignore[arg-type]

    # 8. score_interval (origin_span, fraction axis) with bounds
    ann_score_int = create(
        media_id="s1",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[1, 2]],
        duration=[[1, 4]],
        value=["note"],
        min_quarter=[0, 1],
        max_quarter=[2, 1],
    )
    shifted_si = shift(ann_score_int, [1, 2])
    assert shifted_si.extent.quarter == [[1, 1]]
    assert shifted_si.extent.min_quarter == [1, 2]
    assert shifted_si.extent.max_quarter == [5, 2]

    # 9. text_interval (origin_span, integer axis) with bounds
    ann_text_int = create(
        media_id="txt1",
        payload_type="tag",
        extent_type="text_interval",
        offset=[10],
        length=[5],
        value=["word"],
        min_offset=0,
        max_offset=20,
    )
    shifted_ti = shift(ann_text_int, 5)
    assert shifted_ti.extent.offset == [15]
    assert shifted_ti.extent.min_offset == 5
    assert shifted_ti.extent.max_offset == 25

    # 10. time_interval with bounds
    ann_tint = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.0],
        duration=[2.0],
        value=["v"],
        min_time=0.0,
        max_time=5.0,
    )
    shifted_tint = shift(ann_tint, 1.0)
    assert shifted_tint.extent.time == [2.0]
    assert shifted_tint.extent.min_time == 1.0
    assert shifted_tint.extent.max_time == 6.0

    # 11. text_interval without bounds
    ann_ti_nobounds = create(
        media_id="txt1",
        payload_type="tag",
        extent_type="text_interval",
        offset=[10],
        length=[5],
        value=["word"],
    )
    shifted_ti_nobounds = shift(ann_ti_nobounds, 5)
    assert shifted_ti_nobounds.extent.offset == [15]
    assert shifted_ti_nobounds.extent.min_offset is msgspec.UNSET
    assert shifted_ti_nobounds.extent.max_offset is msgspec.UNSET


def test_union_coverage_branches():
    ann1 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
    )
    # 1. Mismatched bopp_version
    ann_diff_ver = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["b"],
    )
    ann_diff_ver.bopp_version = "0.9"
    with pytest.raises(BoppArgumentError, match="bopp_version"):
        union(ann1, ann_diff_ver)

    # 2. Missing payload on first annotation
    ann_no_payload = Annotation(
        media_id="m1",
        bopp_version="1.0",
        payload=None,  # type: ignore[arg-type]
        extent=Time(time=[1.0], disjoint=True, exhaustive=False),
    )
    with pytest.raises(
        BoppArgumentError, match="First annotation has no valid payload"
    ):
        union(ann_no_payload, ann1)

    # 3. Second operand has time_uniform
    ann_tu = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        value=["x"],
    )
    with pytest.raises(BoppArgumentError, match="is unsupported in union"):
        union(ann1, ann_tu)

    # 4. Confidence mismatch: one has confidence, other does not
    ann_conf = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        time=[2.0],
        value=["b"],
        confidence=[0.8],
    )
    with pytest.raises(BoppArgumentError, match="Confidence type mismatch"):
        union(ann1, ann_conf)
    with pytest.raises(BoppArgumentError, match="Confidence type mismatch"):
        union(ann_conf, ann1)

    # 5. Groupings mismatch: one has groupings, other does not
    ann_group = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["b"],
        groupings=[{"name": "layer1", "ids": [0]}],
    )
    with pytest.raises(BoppArgumentError, match="Groupings mismatch"):
        union(ann1, ann_group)
    with pytest.raises(BoppArgumentError, match="Groupings mismatch"):
        union(ann_group, ann1)

    # 6. Groupings count mismatch
    ann_group_2 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[3.0],
        value=["c"],
        groupings=[
            {"name": "layer1", "ids": [0]},
            {"name": "layer2", "ids": [1]},
        ],
    )
    with pytest.raises(BoppArgumentError, match="Groupings count mismatch"):
        union(ann_group, ann_group_2)

    # 7. Groupings layer name mismatch
    ann_group_diffname = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[3.0],
        value=["c"],
        groupings=[{"name": "other_name", "ids": [0]}],
    )
    with pytest.raises(BoppArgumentError, match="Grouping layer name mismatch"):
        union(ann_group, ann_group_diffname)

    # 8. Conflicting label mapping: label assigned to two different IDs
    ann_lbl1 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
        groupings=[{"name": "g", "ids": [0], "labels": {"verse": 0}}],
    )
    ann_lbl2 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["b"],
        groupings=[{"name": "g", "ids": [1], "labels": {"verse": 1}}],
    )
    with pytest.raises(BoppArgumentError, match="Conflicting label mapping"):
        union(ann_lbl1, ann_lbl2)

    # 9. Confidence facet concatenation and bounds combination
    ann_c1 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        time=[1.0],
        value=["a"],
        confidence=[0.5],
        min_time=1.0,
        max_time=1.0,
    )
    ann_c2 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        time=[2.0],
        value=["b"],
        confidence=[0.8],
        min_time=2.0,
        max_time=2.0,
    )
    res_conf_union = union(ann_c1, ann_c2)
    assert res_conf_union.confidence.confidence == [0.5, 0.8]
    assert res_conf_union.extent.min_time == 1.0
    assert res_conf_union.extent.max_time == 2.0

    # 10. Fraction extent min/max bounds combination
    ann_sq1 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[1, 4]],
        value=["a"],
        min_quarter=[1, 4],
        max_quarter=[1, 4],
    )
    ann_sq2 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[1, 2]],
        value=["b"],
        min_quarter=[1, 2],
        max_quarter=[1, 2],
    )
    res_sq = union(ann_sq1, ann_sq2)
    assert res_sq.extent.min_quarter == [1, 4]
    assert res_sq.extent.max_quarter == [1, 2]

    # 11. Parents lineage: operand with existing parents and operand with no ID
    ann_with_parents = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
        parents=["parent_123"],
    )
    ann_no_id = Annotation(
        media_id="m1",
        bopp_version="1.0",
        payload=TagPayload(value=["b"]),
        extent=Time(time=[2.0], disjoint=True, exhaustive=False),
    )
    with pytest.warns(UserWarning, match="Unioning an annotation with no ID"):
        res_parents = union(ann_with_parents, ann_no_id)
    assert "parent_123" in res_parents.parents
    assert ann_with_parents.id in res_parents.parents

    # 12. Pure payload annotations without extent (extent is UNSET)
    ann_pure1 = create(media_id="m1", payload_type="tag", value=["a"])
    ann_pure2 = create(media_id="m1", payload_type="tag", value=["b"])
    res_pure = union(ann_pure1, ann_pure2)
    assert res_pure.payload.value == ["a", "b"]
    assert res_pure.extent is msgspec.UNSET

    # 13. Union with repeated operand (ID already in parents)
    res_dup = union(ann1, ann1)
    assert len(res_dup.parents) == 1

    # 14. Conflicting label-to-id mapping in grouping
    ann_lid1 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
        groupings=[{"name": "g", "ids": [0], "labels": {"lbl_a": 0}}],
    )
    ann_lid2 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["b"],
        groupings=[{"name": "g", "ids": [0], "labels": {"lbl_b": 0}}],
    )
    with pytest.raises(BoppArgumentError, match="Conflicting label mapping"):
        union(ann_lid1, ann_lid2)

    # 15. Groupings without labels
    ann_nolbl1 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
        groupings=[{"name": "g", "ids": [0]}],
    )
    ann_nolbl2 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["b"],
        groupings=[{"name": "g", "ids": [1]}],
    )
    res_nolbl = union(ann_nolbl1, ann_nolbl2)
    assert res_nolbl.groupings[0].ids == [0, 1]
    assert res_nolbl.groupings[0].labels is msgspec.UNSET

    # 16. Operands sharing a parent ID
    ann_p1 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
        parents=["common_parent"],
    )
    ann_p2 = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["b"],
        parents=["common_parent"],
    )
    res_shared_p = union(ann_p1, ann_p2)
    assert res_shared_p.parents.count("common_parent") == 1


def test_to_uniform_times_coverage_branches():
    # 1. Missing extent
    ann_no_extent = create(
        media_id="m1",
        payload_type="tag",
        value=["a"],
    )
    with pytest.raises(
        BoppArgumentError, match="requires an annotation with a temporal extent"
    ):
        to_uniform_times(ann_no_extent, sampling_period=1.0)

    # 2. Incompatible extent (pixel)
    ann_pixel = create(
        media_id="m1",
        payload_type="tag",
        extent_type="pixel",
        x=[1.0],
        y=[1.0],
        value=["a"],
    )
    with pytest.raises(
        BoppArgumentError, match="is incompatible with to_uniform_times"
    ):
        to_uniform_times(ann_pixel, sampling_period=1.0)

    # 3. Conversion from time_uniform to time_uniform
    ann_tu = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        value=["a", "b", "c"],
    )
    res_tu = to_uniform_times(ann_tu, sampling_period=0.5)
    assert res_tu.extent.sampling_period == 0.5
    assert len(res_tu.payload.value) == 5

    # 4. Empty annotation with explicit start/end time, and without them
    ann_empty = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[],
        value=[],
    )
    with pytest.raises(
        BoppArgumentError, match="Cannot infer temporal bounds from empty annotation"
    ):
        to_uniform_times(ann_empty, sampling_period=1.0)

    res_empty = to_uniform_times(
        ann_empty, sampling_period=1.0, start_time=0.0, end_time=2.0, fill_value="gap"
    )
    assert res_empty.payload.value == ["gap", "gap", "gap"]

    # 5. Multi-column payload with dict fill_value and scalar fill_value error
    ann_mood = create(
        media_id="m1",
        payload_type="mood_thayer",
        extent_type="time",
        time=[1.0],
        valence=[0.5],
        arousal=[0.5],
    )
    with pytest.raises(
        BoppArgumentError, match="Multi-column payload requires fill_value to be a dict"
    ):
        to_uniform_times(
            ann_mood, sampling_period=0.5, start_time=0.0, end_time=2.0, fill_value=0.0
        )

    # Missing fill value in dict
    with pytest.raises(
        BoppArgumentError, match="Missing fill value for payload field 'arousal'"
    ):
        to_uniform_times(
            ann_mood,
            sampling_period=0.5,
            start_time=0.0,
            end_time=2.0,
            fill_value={"valence": 0.0},
        )

    # Valid dict fill value for mood
    res_mood = to_uniform_times(
        ann_mood,
        sampling_period=1.0,
        start_time=0.0,
        end_time=2.0,
        fill_value={"valence": 0.0, "arousal": 0.0},
    )
    assert res_mood.payload.valence == [0.0, 0.5, 0.5]

    # 6. Invalid grouping fill value in dict
    ann_with_g = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
        groupings=[{"name": "sec", "ids": [0]}],
    )
    with pytest.raises(
        BoppValidationError, match="Fill value for grouping 'sec' must be an integer"
    ):
        to_uniform_times(
            ann_with_g,
            sampling_period=1.0,
            start_time=0.0,
            end_time=2.0,
            fill_value={"value": "gap", "sec": "not_an_int"},  # type: ignore[dict-item]
        )

    # 7. Interval extent with overlap='first'
    ann_int = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 1.0],
        duration=[1.0, 1.0],
        value=["first_int", "second_int"],
    )
    # Touching at t=1.0: overlap='first' should pick 'first_int'
    res_first = to_uniform_times(ann_int, sampling_period=1.0, overlap="first")
    assert res_first.payload.value[1] == "first_int"

    # 8. Point extent method='previous' with timestamp before first point (gap)
    ann_pt = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["solo"],
    )
    res_prev_gap = to_uniform_times(
        ann_pt,
        sampling_period=1.0,
        start_time=0.0,
        end_time=3.0,
        fill_value="gap",
        method="previous",
    )
    assert res_prev_gap.payload.value == ["gap", "gap", "solo", "solo"]

    # 9. Confidence facet resampling with gap (and validation)
    ann_conf = create(
        media_id="m1",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        time=[1.0],
        value=["a"],
        confidence=[0.9],
    )
    res_conf_tu = to_uniform_times(
        ann_conf,
        sampling_period=1.0,
        start_time=0.0,
        end_time=2.0,
        fill_value={"value": "gap", "confidence": 0.1},
    )
    assert res_conf_tu.confidence.confidence == [0.1, 0.9, 0.9]

    # 10. Single column payload with dict fill_value using "value"
    res_val_only = to_uniform_times(
        ann_pt,
        sampling_period=1.0,
        start_time=0.0,
        end_time=3.0,
        fill_value={"value": "gap"},
        method="previous",
    )
    assert res_val_only.payload.value == ["gap", "gap", "solo", "solo"]

    # 11. Valid grouping fill value in dict
    res_g_val = to_uniform_times(
        ann_with_g,
        sampling_period=1.0,
        start_time=0.0,
        end_time=2.0,
        fill_value={"value": "gap", "sec": 99},
    )
    assert res_g_val.groupings[0].ids == [99, 0, 0]

    # 12. Interval extent with overlap='latest'
    res_latest = to_uniform_times(ann_int, sampling_period=1.0, overlap="latest")
    assert res_latest.payload.value[1] == "second_int"

    # 13. Confidence not specified in dict fill_value (defaults to 0.0)
    res_conf_def = to_uniform_times(
        ann_conf,
        sampling_period=1.0,
        start_time=0.0,
        end_time=2.0,
        fill_value={"value": "gap"},
    )
    assert res_conf_def.confidence.confidence == [0.0, 0.9, 0.9]

    # 14. Grouping not specified in dict fill_value (defaults to -1)
    res_g_unfilled = to_uniform_times(
        ann_with_g,
        sampling_period=1.0,
        start_time=0.0,
        end_time=2.0,
        fill_value={"value": "gap"},
    )
    assert res_g_unfilled.groupings[0].ids == [-1, 0, 0]


def test_to_pixels_coverage_branches():
    # 1. Empty query points
    ann_box = create(
        media_id="img_1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0],
        width=[10.0],
        y=[0.0],
        height=[10.0],
        value=["box"],
    )
    res_empty_query = to_pixels(ann_box, pixels=[])
    assert res_empty_query.extent.x == []
    assert res_empty_query.extent.y == []
    assert res_empty_query.payload.value == []

    # 2. Empty source annotation
    ann_empty_src = create(
        media_id="img_1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[],
        width=[],
        y=[],
        height=[],
        value=[],
    )
    res_empty_src = to_pixels(ann_empty_src, pixels=[(5.0, 5.0)], fill_value="gap")
    assert res_empty_src.payload.value == ["gap"]

    # 3. Global annotation without extent (broadcast)
    ann_global = create(
        media_id="img_1",
        payload_type="tag",
        value=["global_tag"],
    )
    res_global = to_pixels(ann_global, pixels=[(1.0, 2.0), (3.0, 4.0)])
    assert res_global.payload.value == ["global_tag", "global_tag"]
    assert res_global.extent.x == [1.0, 3.0]
    assert res_global.extent.y == [2.0, 4.0]

    # 4. Point pixel extent with overlap='multiple', 'first', 'latest'
    ann_px_dup = create(
        media_id="img_1",
        payload_type="tag",
        extent_type="pixel",
        x=[5.0, 5.0],
        y=[5.0, 5.0],
        value=["first_val", "second_val"],
    )
    res_px_first = to_pixels(ann_px_dup, pixels=[(5.0, 5.0)], overlap="first")
    assert res_px_first.payload.value == ["first_val"]

    res_px_latest = to_pixels(ann_px_dup, pixels=[(5.0, 5.0)], overlap="latest")
    assert res_px_latest.payload.value == ["second_val"]

    res_px_mult = to_pixels(ann_px_dup, pixels=[(5.0, 5.0)], overlap="multiple")
    assert res_px_mult.payload.value == ["first_val", "second_val"]

    # 5. Multi-column payload with dict fill_value and scalar fill_value error
    ann_mood_box = create(
        media_id="img_1",
        payload_type="mood_thayer",
        extent_type="pixel_box",
        x=[0.0],
        width=[10.0],
        y=[0.0],
        height=[10.0],
        valence=[0.8],
        arousal=[0.4],
    )
    with pytest.raises(
        BoppArgumentError, match="Multi-column payload requires fill_value to be a dict"
    ):
        to_pixels(ann_mood_box, pixels=[(20.0, 20.0)], fill_value=0.0)

    with pytest.raises(
        BoppArgumentError, match="Missing fill value for payload field 'arousal'"
    ):
        to_pixels(
            ann_mood_box,
            pixels=[(20.0, 20.0)],
            fill_value={"valence": 0.0},
        )

    res_mood_box = to_pixels(
        ann_mood_box,
        pixels=[(5.0, 5.0), (20.0, 20.0)],
        fill_value={"valence": 0.0, "arousal": 0.0},
    )
    assert res_mood_box.payload.valence == [0.8, 0.0]

    # 6. Invalid grouping fill value in dict
    ann_box_g = create(
        media_id="img_1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0],
        width=[10.0],
        y=[0.0],
        height=[10.0],
        value=["a"],
        groupings=[{"name": "obj", "ids": [1]}],
    )
    with pytest.raises(
        BoppValidationError, match="Fill value for grouping 'obj' must be an integer"
    ):
        to_pixels(
            ann_box_g,
            pixels=[(50.0, 50.0)],
            fill_value={"value": "gap", "obj": "not_an_int"},  # type: ignore[dict-item]
        )

    # 7. Confidence facet resampling with gap in to_pixels
    ann_box_conf = create(
        media_id="img_1",
        payload_type="tag",
        extent_type="pixel_box",
        confidence_type="likelihood",
        x=[0.0],
        width=[10.0],
        y=[0.0],
        height=[10.0],
        value=["a"],
        confidence=[0.95],
    )
    res_box_conf = to_pixels(
        ann_box_conf,
        pixels=[(5.0, 5.0), (50.0, 50.0)],
        fill_value={"value": "gap", "confidence": 0.05},
    )
    assert res_box_conf.confidence.confidence == [0.95, 0.05]

    # 8. Single-column payload with dict fill_value using 'value'
    res_box_val = to_pixels(ann_box, pixels=[(50.0, 50.0)], fill_value={"value": "gap"})
    assert res_box_val.payload.value == ["gap"]

    # 9. Valid grouping fill value in dict
    res_box_g_val = to_pixels(
        ann_box_g,
        pixels=[(50.0, 50.0)],
        fill_value={"value": "gap", "obj": 99},
    )
    assert res_box_g_val.groupings[0].ids == [99]

    # 10. Pixel box extent with overlap='first'
    ann_boxes_overlap = create(
        media_id="img_1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0, 0.0],
        width=[10.0, 10.0],
        y=[0.0, 0.0],
        height=[10.0, 10.0],
        value=["first_box", "second_box"],
    )
    res_box_first = to_pixels(ann_boxes_overlap, pixels=[(5.0, 5.0)], overlap="first")
    assert res_box_first.payload.value == ["first_box"]

    # 11. Confidence not specified in dict fill_value (defaults to 0.0)
    res_box_conf_def = to_pixels(
        ann_box_conf,
        pixels=[(50.0, 50.0)],
        fill_value={"value": "gap"},
    )
    assert res_box_conf_def.confidence.confidence == [0.0]

    # 12. Grouping not specified in dict fill_value (defaults to -1)
    res_px_g_unfilled = to_pixels(
        ann_box_g,
        pixels=[(50.0, 50.0)],
        fill_value={"value": "gap"},
    )
    assert res_px_g_unfilled.groupings[0].ids == [-1]


def test_copilot_union_and_resampling_coverage(monkeypatch):
    from bopp.core import create
    from bopp.exceptions import BoppArgumentError
    from bopp.transforms import to_pixels, to_times, to_uniform_times, union

    # 1. Union payload column mismatch (optional column voiced)
    ann_freq1 = create(
        media_id="t1",
        payload_type="frequency_hz",
        extent_type="time",
        time=[1.0],
        value=[440.0],
        voiced=[True],
    )
    ann_freq2 = create(
        media_id="t1",
        payload_type="frequency_hz",
        extent_type="time",
        time=[2.0],
        value=[880.0],
    )
    with pytest.raises(BoppArgumentError, match="Payload column mismatch"):
        union(ann_freq1, ann_freq2)

    # 2. Union payload scalar mismatch (ext_schema)
    ann_ext1 = create(
        media_id="t1",
        payload_type="ext",
        ext_schema="schema_a",
        extent_type="time",
        time=[1.0],
        value=["v1"],
    )
    ann_ext2 = create(
        media_id="t1",
        payload_type="ext",
        ext_schema="schema_b",
        extent_type="time",
        time=[2.0],
        value=["v2"],
    )
    with pytest.raises(
        BoppArgumentError, match="Scalar payload field 'ext_schema' must match"
    ):
        union(ann_ext1, ann_ext2)

    # 3. Union confidence column mismatch
    ann_conf1 = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        time=[1.0],
        value=["a"],
        confidence=[0.9],
        num_annotators=[5],
    )
    ann_conf2 = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        time=[2.0],
        value=["b"],
        confidence=[0.8],
    )
    with pytest.raises(BoppArgumentError, match="Confidence column mismatch"):
        union(ann_conf1, ann_conf2)

    # 4. Union confidence scalar mismatch
    ann_conf_s1 = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        time=[1.0],
        value=["a"],
        confidence=[0.9],
        num_annotators_common=5,
    )
    ann_conf_s2 = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        time=[2.0],
        value=["b"],
        confidence=[0.8],
        num_annotators_common=10,
    )
    with pytest.raises(
        BoppArgumentError,
        match="Scalar confidence field 'num_annotators_common' must match",
    ):
        union(ann_conf_s1, ann_conf_s2)

    # 5. Fallback observation count via groupings when source_cols is empty
    ann_global = create(
        media_id="t1",
        payload_type="tag",
        value=["g1"],
        groupings=[{"name": "layer1", "ids": [0]}],
    )
    with monkeypatch.context() as m:
        # Simulate payload and confidence having no list columns
        m.setattr("bopp.transforms._get_facet_list_fields", lambda s: {})
        res_t = to_times(ann_global, times=[0.0])
        assert len(res_t.groupings[0].ids) == 1

        ann_tu = create(
            media_id="t1",
            payload_type="tag",
            extent_type="time_uniform",
            sampling_period=1.0,
            start_time=0.0,
            value=["g1"],
            groupings=[{"name": "layer1", "ids": [0]}],
        )
        res_ut = to_uniform_times(
            ann_tu, sampling_period=1.0, start_time=0.0, end_time=0.0
        )
        assert len(res_ut.groupings[0].ids) == 1

        res_px = to_pixels(ann_global, pixels=[(0.0, 0.0)])
        assert len(res_px.groupings[0].ids) == 1


def test_more_coverage_remaining_branches():
    from bopp.core import create
    from bopp.exceptions import BoppArgumentError
    from bopp.transforms import filter_by, to_times, trim

    # 1. trim invalid target field on time_uniform
    ann_tu = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time_uniform",
        sampling_period=1.0,
        start_time=0.0,
        value=["a", "b", "c"],
    )
    with pytest.raises(BoppArgumentError, match="Unsupported target_field"):
        trim(ann_tu, start=1.0, target_field="pitch")

    # 2. trim on time_uniform with only max_time set
    ann_tu_max = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time_uniform",
        sampling_period=1.0,
        start_time=0.0,
        max_time=10.0,
        value=["a", "b"],
    )
    trimmed_max = trim(ann_tu_max, end=5.0)
    assert getattr(trimmed_max.extent, "max_time", None) == 5.0

    # trim on time_uniform without min_time or max_time set
    trimmed_tu = trim(ann_tu, end=1.5)
    assert len(trimmed_tu.payload.value) == 2

    # 3. filter_by on time_uniform
    with pytest.raises(
        BoppArgumentError, match="filter_by is not supported for 'time_uniform'"
    ):
        filter_by(ann_tu, predicate=lambda obs: True)

    # 4. to_times on time_uniform with sample_rate
    res_sr = to_times(ann_tu, sample_rate=2.0)
    assert len(res_sr.payload.value) == 5

    # 5. to_times on time_uniform with times and method previous
    res_prev = to_times(ann_tu, times=[0.5, 1.5], method="previous")
    assert res_prev.payload.value == ["a", "b"]


def test_copilot_bounds_and_drift_fixes():
    from bopp.core import create
    from bopp.exceptions import BoppArgumentError
    from bopp.transforms import to_pixels, to_times, to_uniform_times, union

    # 1. Union aggregating effective bounds from unbounded operands
    ann1 = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[1.0],
        min_time=0.0,
        max_time=1.0,
        exhaustive=True,
        disjoint=True,
        value=["a"],
    )
    assert ann1.extent.exhaustive is True

    ann2 = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time_interval",
        time=[10.0],
        duration=[1.0],
        value=["b"],
    )
    u_ann = union(ann1, ann2)
    assert getattr(u_ann.extent, "min_time", None) == 0.0
    assert getattr(u_ann.extent, "max_time", None) == 11.0
    assert u_ann.extent.exhaustive is False

    # 2. Recomputing grouping disjointness after resampling in to_pixels and to_times
    ann_group = create(
        media_id="t1",
        payload_type="tag",
        extent_type="pixel",
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        value=["a", "b"],
        groupings=[{"name": "layer1", "ids": [0, 0], "disjoint": True}],
    )
    res_px_dup = to_pixels(ann_group, pixels=[(0.0, 0.0), (0.0, 0.0)])
    assert res_px_dup.extent.disjoint is False
    assert res_px_dup.groupings[0].disjoint is False

    ann_t_group = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        time=[0.0, 1.0],
        value=["a", "b"],
        groupings=[{"name": "layer1", "ids": [0, 0], "disjoint": True}],
    )
    res_t_dup = to_times(ann_t_group, times=[0.0, 0.0])
    assert res_t_dup.extent.disjoint is False
    assert res_t_dup.groupings[0].disjoint is False

    # 3. Marker resampling without column carriers rejected
    ann_marker = create(
        media_id="t1",
        payload_type="marker",
        extent_type="time",
        time=[0.0, 2.0],
        description="label",
    )
    with pytest.raises(
        BoppArgumentError,
        match="to_uniform_times requires at least one columnar field",
    ):
        to_uniform_times(ann_marker, sampling_period=1.0)

    # 4. Reject non-finite bounds in to_uniform_times
    ann_time = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time",
        time=[0.0, 1.0],
        value=["a", "b"],
    )
    with pytest.raises(BoppArgumentError, match="must be finite numbers"):
        to_uniform_times(ann_time, sampling_period=1.0, start_time=float("nan"))
    with pytest.raises(BoppArgumentError, match="must be finite numbers"):
        to_uniform_times(ann_time, sampling_period=1.0, end_time=float("inf"))

    # 5. Floating point endpoint drift in to_uniform_times
    ann_drift = create(
        media_id="t1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[0.3],
        value=["drift"],
    )
    res_drift = to_uniform_times(ann_drift, sampling_period=0.1)
    # 0.0, 0.1, 0.2, 0.3 should all be matched to "drift" without gap/fill_value
    assert res_drift.payload.value == ["drift", "drift", "drift", "drift"]

    # 6. Global resampling with multiple rows under overlap strategies
    ann_global_multi = create(
        media_id="t1",
        payload_type="tag",
        value=["a", "b"],
    )
    # to_times
    res_gt_first = to_times(ann_global_multi, times=[1.0, 2.0], overlap="first")
    assert res_gt_first.payload.value == ["a", "a"]
    res_gt_latest = to_times(ann_global_multi, times=[1.0, 2.0], overlap="latest")
    assert res_gt_latest.payload.value == ["b", "b"]
    res_gt_multi = to_times(ann_global_multi, times=[1.0, 2.0], overlap="multiple")
    assert res_gt_multi.payload.value == ["a", "b", "a", "b"]
    assert res_gt_multi.extent.time == [1.0, 1.0, 2.0, 2.0]

    # to_pixels
    res_gp_first = to_pixels(
        ann_global_multi, pixels=[(0.0, 0.0), (1.0, 1.0)], overlap="first"
    )
    assert res_gp_first.payload.value == ["a", "a"]
    res_gp_latest = to_pixels(
        ann_global_multi, pixels=[(0.0, 0.0), (1.0, 1.0)], overlap="latest"
    )
    assert res_gp_latest.payload.value == ["b", "b"]
    res_gp_multi = to_pixels(
        ann_global_multi, pixels=[(0.0, 0.0), (1.0, 1.0)], overlap="multiple"
    )
    assert res_gp_multi.payload.value == ["a", "b", "a", "b"]
    assert res_gp_multi.extent.x == [0.0, 0.0, 1.0, 1.0]
    assert res_gp_multi.extent.y == [0.0, 0.0, 1.0, 1.0]
