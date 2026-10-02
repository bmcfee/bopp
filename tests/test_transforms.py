import copy
import warnings

import msgspec
import pytest

import bopp
from bopp.core import create
from bopp.exceptions import BoppArgumentError
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.extent.time_frequency_box import TimeFrequencyBoxExtent
from bopp.models.v1.metadata.human import HumanAnnotationMetadata
from bopp.models.v1.payload.tag_open import TagOpenPayload
from bopp.transforms import _get_list_fields_with_length, trim


def test_get_list_fields_with_length_empty():
    class EmptyStruct(msgspec.Struct):
        field_a: int = 1
        field_b: str = "test"

    s = EmptyStruct()
    assert _get_list_fields_with_length(s, 2) == {}


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
    # Boxes: [100..300] (keep, clipped to [250..300]), [200..400] (keep, clipped to [250..400]),
    #        [500..800] (keep, clipped to [500..600]), [900..1000] (dropped: p_min >= end)
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
    # Targets min_max axis directly with items falling below start, above end, and inside range
    extent = TimeFrequencyBoxExtent(
        time=[0.0, 0.0, 0.0],
        duration=[1.0, 1.0, 1.0],
        freq_min=[50.0, 200.0, 500.0],
        freq_max=[100.0, 400.0, 800.0],
    )
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
        extent=extent,
        payload=TagOpenPayload(value=["low", "mid", "high"]),
    )
    bopp.validate_and_set_annotation_id(ann)

    # Non-strict trim to range [150.0, 600.0]
    # Item 0 ([50, 100]): p_max <= start (100 <= 150) -> skipped (hits line 222)
    # Item 1 ([200, 400]): fully inside
    # Item 2 ([500, 800]): overlaps upper bound -> clipped to [500, 600]
    trimmed_non_strict = trim(ann, start=150.0, end=600.0, target_field="frequency", strict=False)
    assert trimmed_non_strict.extent.freq_min == [200.0, 500.0]
    assert trimmed_non_strict.extent.freq_max == [400.0, 600.0]
    assert trimmed_non_strict.payload.value == ["mid", "high"]

    # Strict trim to range [150.0, 600.0]
    # Item 0 ([50, 100]): p_min < start -> skipped in strict mode
    # Item 1 ([200, 400]): fully inside -> kept
    # Item 2 ([500, 800]): p_max > end -> skipped in strict mode
    trimmed_strict = trim(ann, start=150.0, end=600.0, target_field="frequency", strict=True)
    assert trimmed_strict.extent.freq_min == [200.0]
    assert trimmed_strict.extent.freq_max == [400.0]
    assert trimmed_strict.payload.value == ["mid"]


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
