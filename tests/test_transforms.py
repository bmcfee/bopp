import copy
import warnings

import msgspec
import pytest

import bopp
from bopp.core import create
from bopp.exceptions import BoppArgumentError, BoppValidationError
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.confidence.likelihood import LikelihoodConfidence
from bopp.models.v1.extent.time_frequency_box import TimeFrequencyBoxExtent
from bopp.models.v1.extent.times import Times
from bopp.models.v1.metadata.derived import DerivedAnnotationMetadata
from bopp.models.v1.metadata.human import HumanAnnotationMetadata
from bopp.models.v1.payload.mood_thayer import MoodThayerPayload
from bopp.models.v1.payload.tag_open import TagOpenPayload
from bopp.transforms import (
    FilterRecord,
    _derive_parents_and_sandbox,
    _get_facet_list_fields,
    _get_list_fields_with_length,
    _rebuild_annotation,
    _resample_struct,
    _subset_struct_lists,
    filter_by,
    to_times,
    trim,
)


def test_get_list_fields_with_length_empty():
    class EmptyStruct(msgspec.Struct):
        field_a: int = 1
        field_b: str = "test"

    s = EmptyStruct()
    assert _get_list_fields_with_length(s, 2) == {}


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
        payload=TagOpenPayload(value=["pop"]),
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
        payload=TagOpenPayload(value=["pop"]),
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
        payload=TagOpenPayload(value=["test"]),
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
    subsetted = _subset_struct_lists(sample, [0, 2], expected_length=3)
    assert subsetted.tags == ["a", "c"]
    assert subsetted.scores == [1.0, 3.0]
    assert subsetted.extra_scalar == 42

    # With overrides
    overridden = _subset_struct_lists(
        sample, [1], expected_length=3, overrides={"tags": ["custom"]}
    )
    assert overridden.tags == ["custom"]
    assert overridden.scores == [2.0]


def test_resample_struct_helper():
    assert _resample_struct(None, {}, [], {}) is None
    assert _resample_struct(msgspec.UNSET, {}, [], {}) is msgspec.UNSET

    payload = TagOpenPayload(value=["apple", "banana"])
    cols = _get_facet_list_fields(payload)
    resampled = _resample_struct(
        payload, cols, [1, None, 0], fill_map={"value": "empty"}
    )
    assert resampled.value == ["banana", "empty", "apple"]


def test_rebuild_annotation_helper():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        id="old_id",
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Times(time=[1.0]),
        payload=TagOpenPayload(value=["test"]),
    )
    rebuilt = _rebuild_annotation(
        ann,
        parents=["old_id"],
        sandbox={"foo": "bar"},
        payload=TagOpenPayload(value=["new_test"]),
        extent=Times(time=[2.0]),
    )
    assert rebuilt.id != "old_id"
    assert rebuilt.parents == ["old_id"]
    assert rebuilt.sandbox == {"foo": "bar"}
    assert rebuilt.payload.value == ["new_test"]
    assert rebuilt.extent.time == [2.0]


def test_trim_invalid_arguments():
    ann = create(
        media_id="test_media",
        payload_kind="tag_open",
        extent_kind="time",
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


def test_trim_warning_no_id():
    metadata = HumanAnnotationMetadata(annotator_id="user_1", tool="manual")
    ann = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagOpenPayload(value=["pop"]),
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
        payload=TagOpenPayload(value=["a", "b"]),
    )
    bopp.validate_and_set_annotation_id(ann)

    with pytest.raises(BoppArgumentError, match="has no valid schema tag"):
        trim(ann, start=1.0, end=2.0)


def test_trim_no_extent():
    metadata = HumanAnnotationMetadata(annotator_id="user_1", tool="manual")
    ann = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagOpenPayload(value=["pop"]),
    )
    bopp.validate_and_set_annotation_id(ann)

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
        payload_kind="tag_open",
        extent_kind="time",
        confidence_kind="likelihood",
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


def test_trim_point_extent_open_ended_bounds():
    """Verify trim on point extent with only start, only end, and check boundary conditions."""
    ann = create(
        media_id="test_media",
        payload_kind="tag_open",
        extent_kind="time",
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
        payload_kind="tag_open",
        extent_kind="time_interval",
        confidence_kind="likelihood",
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
        payload_kind="tag_open",
        extent_kind="time_interval",
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
        ),
        payload=TagOpenPayload(value=["low", "high"]),
    )
    bopp.validate_and_set_annotation_id(ann)

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
    )
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=extent,
        payload=TagOpenPayload(value=["low", "mid", "high", "ultra"]),
    )
    bopp.validate_and_set_annotation_id(ann)

    # Trim frequency axis non-strict [250..600]
    trimmed_freq = trim(ann, start=250.0, end=600.0, target_field="frequency", reset=True)
    assert trimmed_freq.extent.freq_min == [0.0, 0.0, 250.0]
    assert trimmed_freq.extent.freq_max == [50.0, 150.0, 350.0]
    assert trimmed_freq.payload.value == ["low", "mid", "high"]

    # Trim frequency axis strict (150 to 450 strictly contains [200, 400])
    trimmed_freq_strict = trim(ann, start=150.0, end=450.0, target_field="frequency", strict=True)
    assert trimmed_freq_strict.extent.freq_min == [200.0]
    assert trimmed_freq_strict.extent.freq_max == [400.0]
    assert trimmed_freq_strict.payload.value == ["mid"]


def test_trim_min_max_strict_and_non_strict():
    extent = TimeFrequencyBoxExtent(
        time=[0.0, 0.0, 0.0, 0.0],
        duration=[1.0, 1.0, 1.0, 1.0],
        freq_min=[50.0, 200.0, 500.0, 700.0],
        freq_max=[100.0, 400.0, 800.0, 900.0],
    )
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=extent,
        payload=TagOpenPayload(value=["low", "mid", "high", "ultra"]),
    )
    bopp.validate_and_set_annotation_id(ann)

    # Non-strict trim with start and end
    trimmed_non_strict = trim(ann, start=150.0, end=600.0, target_field="frequency", strict=False)
    assert trimmed_non_strict.extent.freq_min == [200.0, 500.0]
    assert trimmed_non_strict.extent.freq_max == [400.0, 600.0]
    assert trimmed_non_strict.payload.value == ["mid", "high"]

    # Non-strict trim with only start (tests start is not None, end is None branch)
    trimmed_only_start_ns = trim(ann, start=300.0, target_field="frequency", strict=False)
    assert trimmed_only_start_ns.extent.freq_min == [300.0, 500.0, 700.0]
    assert trimmed_only_start_ns.extent.freq_max == [400.0, 800.0, 900.0]

    # Non-strict trim with only end (tests start is None, end is not None branch)
    trimmed_only_end_ns = trim(ann, end=450.0, target_field="frequency", strict=False)
    assert trimmed_only_end_ns.extent.freq_min == [50.0, 200.0]
    assert trimmed_only_end_ns.extent.freq_max == [100.0, 400.0]

    # Strict trim with start and end
    trimmed_strict = trim(ann, start=150.0, end=600.0, target_field="frequency", strict=True)
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
    )
    ann_empty = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=extent_empty,
        payload=TagOpenPayload(value=[]),
    )
    bopp.validate_and_set_annotation_id(ann_empty)

    trimmed_empty = trim(ann_empty, start=100.0, end=200.0, target_field="frequency", strict=True)
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
    )
    ann_oob = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=extent_out_of_bounds,
        payload=TagOpenPayload(value=["low1", "low2"]),
    )
    bopp.validate_and_set_annotation_id(ann_oob)

    # Trim frequency axis [100.0, 200.0] - all observations should be filtered out
    trimmed_oob = trim(ann_oob, start=100.0, end=200.0, target_field="frequency", strict=False)
    assert trimmed_oob.extent.freq_min == []
    assert trimmed_oob.extent.freq_max == []
    assert trimmed_oob.payload.value == []
    assert trimmed_oob.parents == [ann_oob.id]


def test_trim_pixel_box():
    ann = create(
        media_id="test_image",
        payload_kind="tag_open",
        extent_kind="pixel_box",
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
        payload_kind="tag_open",
        extent_kind="time",
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
        payload_kind="tag_open",
        extent_kind="time",
        confidence_kind="likelihood",
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
        payload_kind="tag_open",
        extent_kind="time",
        time=[1.0, 2.0, 3.0],
        value=["rock", "pop", "metal"],
    )

    filtered = filter_by(ann, lambda v: len(v) == 4, target="value")
    assert filtered.payload.value == ["rock"]
    assert filtered.extent.time == [1.0]


def test_filter_by_target_on_extent():
    ann = create(
        media_id="test_media",
        payload_kind="tag_open",
        extent_kind="time",
        confidence_kind="likelihood",
        time=[1.0, 2.0, 3.0, 4.0],
        value=["a", "b", "c", "d"],
        confidence=[0.1, 0.2, 0.3, 0.4],
    )

    filtered = filter_by(ann, lambda t: t > 2.0, facet="extent", target="time")
    assert filtered.extent.time == [3.0, 4.0]
    assert filtered.payload.value == ["c", "d"]
    assert filtered.confidence.confidence == [0.3, 0.4]


def test_filter_by_confidence_facet():
    ann = create(
        media_id="test_media",
        payload_kind="tag_open",
        extent_kind="time",
        confidence_kind="likelihood",
        time=[1.0, 2.0, 3.0],
        value=["a", "b", "c"],
        confidence=[0.2, 0.8, 0.5],
    )

    filtered = filter_by(ann, lambda c: c >= 0.5, facet="confidence", target="confidence")
    assert filtered.confidence.confidence == [0.8, 0.5]
    assert filtered.payload.value == ["b", "c"]
    assert filtered.extent.time == [2.0, 3.0]


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
    bopp.validate_and_set_annotation_id(ann)

    # Filter with record
    filtered = filter_by(ann, lambda r: r.valence > 0 and r.arousal > 0)
    assert filtered.payload.valence == [0.5]
    assert filtered.payload.arousal == [0.1]

    # Explicit target on one field
    filtered_v = filter_by(ann, lambda v: v > 0, target="valence")
    assert filtered_v.payload.valence == [0.5, 0.8]
    assert filtered_v.payload.arousal == [0.1, -0.3]


def test_filter_by_facet_all():
    ann = create(
        media_id="test_media",
        payload_kind="tag_open",
        extent_kind="time",
        confidence_kind="likelihood",
        time=[1.0, 2.0, 3.0, 4.0],
        value=["intro", "verse", "chorus", "verse"],
        confidence=[0.9, 0.4, 0.95, 0.85],
    )

    # Filter across facets using record
    filtered = filter_by(
        ann,
        lambda r: r.confidence >= 0.8 and r.value.startswith("v"),
        facet="all",
    )
    assert filtered.payload.value == ["verse"]
    assert filtered.extent.time == [4.0]
    assert filtered.confidence.confidence == [0.85]

    # Filter across facets specifying target
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
        payload=TagOpenPayload(value=["one", "two", "three"]),
    )
    filtered = filter_by(ann, lambda v: v != "two")
    assert filtered.payload.value == ["one", "three"]
    assert filtered.extent is msgspec.UNSET


def test_filter_by_invalid_arguments():
    ann = create(
        media_id="test_media",
        payload_kind="tag_open",
        extent_kind="time",
        time=[1.0, 2.0],
        value=["a", "b"],
    )

    with pytest.raises(BoppArgumentError, match="Invalid facet 'invalid'"):
        filter_by(ann, lambda x: True, facet="invalid")  # type: ignore[arg-type]

    with pytest.raises(BoppArgumentError, match="Facet 'confidence' is not present"):
        filter_by(ann, lambda x: True, facet="confidence")

    with pytest.raises(BoppArgumentError, match="Target field 'unknown' not found in facet 'payload'"):
        filter_by(ann, lambda x: True, target="unknown")

    with pytest.raises(BoppArgumentError, match="Target field 'unknown' not found in any annotation facet"):
        filter_by(ann, lambda x: True, facet="all", target="unknown")


def test_filter_by_no_id_warning():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="test_media",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagOpenPayload(value=["pop"]),
    )

    with pytest.warns(UserWarning, match="Filtering an annotation with no ID"):
        res = filter_by(ann, lambda v: True)

    assert res.parents == []
    assert isinstance(res.metadata, DerivedAnnotationMetadata)
    assert res.metadata.transform == "filter_by"


def test_filter_by_empty_or_none_remaining():
    ann = create(
        media_id="test_media",
        payload_kind="tag_open",
        extent_kind="time",
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
        payload_kind="tag_open",
        extent_kind="time",
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
        payload_kind="tag_open",
        extent_kind="time",
        time=[1.0, 2.0],
        value=["a", "b"],
    )

    # Neither times nor sample_rate
    with pytest.raises(BoppArgumentError, match="Exactly one of 'times' or 'sample_rate'"):
        to_times(ann)

    # Both times and sample_rate
    with pytest.raises(BoppArgumentError, match="Exactly one of 'times' or 'sample_rate'"):
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
        payload_kind="tag_open",
        extent_kind="pixel_box",
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
        payload_kind="tag_open",
        extent_kind="midi_tick",
        tick=[0, 480],
        value=["a", "b"],
    )
    with pytest.raises(BoppArgumentError, match="is incompatible with to_times"):
        to_times(ann_midi, times=[0.0, 1.0])

    ann_score = create(
        media_id="track",
        payload_kind="tag_open",
        extent_kind="score_quarter",
        quarter=[0.0, 1.0],
        value=["a", "b"],
    )
    with pytest.raises(BoppArgumentError, match="is incompatible with to_times"):
        to_times(ann_score, times=[0.0, 1.0])


def test_to_times_from_time_interval_with_sample_rate():
    ann = create(
        media_id="audio",
        payload_kind="tag_open",
        extent_kind="time_interval",
        confidence_kind="likelihood",
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


def test_to_times_overlap_strategies():
    ann = create(
        media_id="audio",
        payload_kind="tag_open",
        extent_kind="time_interval",
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
        payload_kind="tag_open",
        extent_kind="time_interval",
        confidence_kind="likelihood",
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
        payload_kind="tag_open",
        extent_kind="time",
        time=[1.0, 3.0],
        value=["a", "b"],
    )

    # previous lookup
    res_prev = to_times(ann, times=[0.5, 1.5, 3.0, 4.0], method="previous", fill_value="none")
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
        payload=TagOpenPayload(value=["rock"]),
    )
    bopp.validate_and_set_annotation_id(ann)

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
        ),
        payload=TagOpenPayload(value=["low", "high"]),
    )
    bopp.validate_and_set_annotation_id(ann)

    res = to_times(ann, times=[1.0, 3.0])
    assert res.extent.time == [1.0, 3.0]
    assert res.payload.value == ["low", "high"]


def test_to_times_fill_value_dict_with_value_fallback_and_confidence():
    """Verify fill_value dict with 'value' key fallback for single-column payload and confidence fill."""
    ann = create(
        media_id="audio",
        payload_kind="tag_open",
        extent_kind="time_interval",
        confidence_kind="likelihood",
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
        extent=Times(time=[1.0]),
        payload=MoodThayerPayload(valence=[0.5], arousal=[0.2]),
    )
    bopp.validate_and_set_annotation_id(ann)

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
        ),
        payload=MoodThayerPayload(valence=[0.5], arousal=[0.8]),
    )
    bopp.validate_and_set_annotation_id(ann)

    # Scalar fill value for multi-column payload raises error
    with pytest.raises(BoppArgumentError, match="requires fill_value to be a dict"):
        to_times(ann, times=[0.0], fill_value=0.0)

    # Incomplete dict raises error
    with pytest.raises(BoppArgumentError, match="Missing fill value for payload field 'arousal'"):
        to_times(ann, times=[0.0], fill_value={"valence": 0.0})

    # Proper dict fill
    res = to_times(ann, times=[0.0, 1.5], fill_value={"valence": 0.0, "arousal": 0.0})
    assert res.extent.time == [0.0, 1.5]
    assert res.payload.valence == [0.0, 0.5]
    assert res.payload.arousal == [0.0, 0.8]


def test_to_times_multi_column_payload_none_fill_value():
    """Verify multi-column payload with fill_value=None successfully fills missing positions with None."""
    class OptionalFieldPayload(msgspec.Struct, tag_field="payload_type", tag="opt_payload"):
        label: list[str | None]
        comment: list[str | None]

    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Times(time=[1.0]),
        payload=OptionalFieldPayload(label=["test"], comment=["note"]),
    )
    bopp.validate_and_set_annotation_id(ann)

    res = to_times(ann, times=[0.0, 1.0], method="previous", fill_value=None)
    assert res.extent.time == [0.0, 1.0]
    assert res.payload.label == [None, "test"]
    assert res.payload.comment == [None, "note"]


def test_to_times_sample_rate_grid_boundary_pop():
    """Verify target_times grid trimming when step multiple lands precisely on t_max."""
    ann = create(
        media_id="audio",
        payload_kind="tag_open",
        extent_kind="time",
        time=[0.0, 1.0],
        value=["start", "end"],
    )
    # sample_rate=1.0 over [0.0, 1.0] creates points [0.0, 1.0]
    res = to_times(ann, sample_rate=1.0)
    assert res.extent.time == [0.0, 1.0]


def test_to_times_fill_value_type_validation_error():
    ann = create(
        media_id="audio",
        payload_kind="tag_open",
        extent_kind="time_interval",
        time=[1.0],
        duration=[1.0],
        value=["tag"],
    )

    # tag_open requires list[str], so passing an invalid non-string (int) causes validation error
    with pytest.raises(BoppValidationError, match="Invalid fill value"):
        to_times(ann, times=[0.0], fill_value=123)  # type: ignore[arg-type]


def test_to_times_empty_annotation_and_empty_times():
    ann = create(
        media_id="audio",
        payload_kind="tag_open",
        extent_kind="time",
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
        extent=Times(time=[]),
        payload=TagOpenPayload(value=[]),
        confidence=LikelihoodConfidence(confidence=[]),
    )
    bopp.validate_and_set_annotation_id(ann)

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
        extent=Times(time=[]),
        payload=TagOpenPayload(value=[]),
        confidence=LikelihoodConfidence(confidence=[]),
    )
    bopp.validate_and_set_annotation_id(ann)

    with pytest.raises(BoppValidationError, match="Invalid fill value for confidence struct"):
        to_times(ann, times=[0.0, 1.0], fill_value={"value": "silence", "confidence": "not_a_float"})


def test_to_times_point_extent_previous_with_gaps_before_first_observation():
    """Verify that queries before the first timestamp in method='previous' produce gaps and trigger fill."""
    ann = create(
        media_id="audio",
        payload_kind="tag_open",
        extent_kind="time",
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
        payload=TagOpenPayload(value=["v"]),
    )

    with pytest.warns(UserWarning, match="Converting an annotation with no ID"):
        res = to_times(ann, times=[0.0])

    assert res.parents == []

    # Verify immutability
    ann_with_id = create(
        media_id="audio",
        payload_kind="tag_open",
        extent_kind="time",
        time=[1.0],
        value=["val"],
        sandbox={"key": 1},
    )
    ann_copy = copy.deepcopy(ann_with_id)
    res_immut = to_times(ann_with_id, times=[1.0])
    assert ann_with_id == ann_copy
    assert res_immut.sandbox is not ann_with_id.sandbox
