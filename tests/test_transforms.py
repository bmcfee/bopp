import copy

import msgspec
import pytest

import bopp
from bopp.exceptions import BoppArgumentError
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.confidence.likelihood import LikelihoodConfidence
from bopp.models.v1.extent.time_frequency_box import TimeFrequencyBoxExtent
from bopp.models.v1.extent.time_interval import TimeIntervalExtent
from bopp.models.v1.extent.times import Times
from bopp.models.v1.metadata.human import HumanAnnotationMetadata
from bopp.models.v1.payload.tag_open import TagOpenPayload
from bopp.transforms import trim


def test_trim_invalid_arguments():
    ann = bopp.create(media_id="test_media", payload_kind="tag_open", value=["rock"])

    # No start or end provided
    with pytest.raises(BoppArgumentError, match="At least one of 'start' or 'end'"):
        trim(ann)

    # start > end
    with pytest.raises(BoppArgumentError, match="start .* must be <= end"):
        trim(ann, start=10.0, end=5.0)

    # reset=True with start=None
    with pytest.raises(BoppArgumentError, match="reset=True requires 'start'"):
        trim(ann, end=5.0, reset=True)


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
    ann = bopp.create(
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
    ann = bopp.create(
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
    # [0, 2] -> [1, 2] (dur 1)
    # [2, 6] -> [2, 6] (dur 4)
    # [5, 9] -> [5, 6] (dur 1)
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
    ann = bopp.create(
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
    # Box 1: [0, 3] -> clipped to [1, 3] (duration 2). reset_time shifts to [0, 2]
    # Box 2: [4, 9] -> clipped to [4, 5] (duration 1). reset_time shifts to [3, 1]
    assert trimmed.extent.time == [0.0, 3.0]
    assert trimmed.extent.duration == [2.0, 1.0]
    assert trimmed.extent.freq_min == [100.0, 200.0]
    assert trimmed.extent.freq_max == [500.0, 800.0]
    assert trimmed.payload.value == ["low", "high"]


def test_trim_transitive_parents_and_immutability():
    ann1 = bopp.create(
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
