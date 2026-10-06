import builtins
import sys
from unittest.mock import patch

import msgspec
import pytest

import bopp
from bopp.core import create
from bopp.evaluation import (
    _check_mir_eval_available,
    _extract_events,
    _extract_intervals,
    _extract_labels,
    _extract_note_pitches_in_hz,
    evaluate,
)
from bopp.exceptions import BoppArgumentError, BoppError
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.extent.time_interval import TimeIntervalExtent
from bopp.models.v1.extent.times import Times
from bopp.models.v1.metadata.human import HumanAnnotationMetadata
from bopp.models.v1.payload.note_hz import NoteHzPayload
from bopp.models.v1.payload.note_midi import NoteMidiPayload
from bopp.models.v1.payload.pitch_contour_hz import PitchContourPayload
from bopp.models.v1.payload.segment_multi import MultiSegmentPayload
from bopp.models.v1.payload.tag_open import TagOpenPayload
from bopp.models.v1.payload.tempo import TempoPayload


def test_check_mir_eval_available_missing_numpy():
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("No module named 'numpy'")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=mock_import):
        with pytest.raises(BoppError, match="Evaluation requires 'numpy'"):
            _check_mir_eval_available()


def test_check_mir_eval_available_missing_mir_eval():
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mir_eval":
            raise ImportError("No module named 'mir_eval'")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=mock_import):
        with pytest.raises(BoppError, match="Evaluation requires 'mir_eval'"):
            _check_mir_eval_available()


def test_evaluate_onset():
    ref = create(
        media_id="audio",
        payload_kind="onset",
        extent_kind="time",
        time=[1.0, 2.0, 3.0],
        value=[1, 1, 1],
    )
    est = create(
        media_id="audio",
        payload_kind="onset",
        extent_kind="time",
        time=[1.02, 2.01, 3.05],
        value=[1, 1, 1],
    )

    scores = evaluate(ref, est)
    assert "F-measure" in scores
    assert scores["F-measure"] == 1.0


def test_evaluate_onset_from_time_interval():
    ref = create(
        media_id="audio",
        payload_kind="onset",
        extent_kind="time_interval",
        time=[1.0, 2.0],
        duration=[0.5, 0.5],
        value=[1, 1],
    )
    est = create(
        media_id="onset",
        payload_kind="onset",
        extent_kind="time",
        time=[1.01, 2.01],
        value=[1, 1],
    )
    scores = evaluate(ref, est, task="onset")
    assert scores["F-measure"] == 1.0


def test_evaluate_beat():
    ref = create(
        media_id="audio",
        payload_kind="beat",
        extent_kind="time",
        time=[0.0, 0.5, 1.0, 1.5],
        value=[1, 2, 3, 4],
    )
    est = create(
        media_id="audio",
        payload_kind="beat",
        extent_kind="time",
        time=[0.01, 0.49, 1.02, 1.51],
        value=[1, 2, 3, 4],
    )

    scores = evaluate(ref, est)
    assert "F-measure" in scores
    assert scores["F-measure"] > 0.9


def test_evaluate_tempo():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ref = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TempoPayload(value=[120.0, 60.0]),
    )
    est = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TempoPayload(value=[120.0, 60.0]),
    )
    bopp.validate_and_set_annotation_id(ref)
    bopp.validate_and_set_annotation_id(est)

    scores = evaluate(ref, est)
    assert "P-score" in scores
    assert scores["P-score"] == 1.0


def test_evaluate_chord():
    ref = create(
        media_id="audio",
        payload_kind="chord",
        extent_kind="time_interval",
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=["C:maj", "G:maj"],
    )
    est = create(
        media_id="audio",
        payload_kind="chord",
        extent_kind="time_interval",
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=["C:maj", "G:maj"],
    )

    scores = evaluate(ref, est)
    assert "mirex" in scores
    assert scores["mirex"] == 1.0


def test_evaluate_chord_from_point_times():
    # Consecutive point events infer intervals
    ref = create(
        media_id="audio",
        payload_kind="chord",
        extent_kind="time",
        time=[0.0, 2.0, 4.0],
        value=["C:maj", "G:maj", "C:maj"],
    )
    est = create(
        media_id="audio",
        payload_kind="chord",
        extent_kind="time_interval",
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=["C:maj", "G:maj"],
    )

    scores = evaluate(ref, est)
    assert scores["mirex"] == 1.0


def test_evaluate_segment():
    ref = create(
        media_id="audio",
        payload_kind="segment_open",
        extent_kind="time_interval",
        time=[0.0, 4.0],
        duration=[4.0, 6.0],
        value=["verse", "chorus"],
    )
    est = create(
        media_id="audio",
        payload_kind="segment_open",
        extent_kind="time_interval",
        time=[0.0, 4.0],
        duration=[4.0, 6.0],
        value=["verse", "chorus"],
    )

    scores = evaluate(ref, est)
    assert "Pairwise F-measure" in scores
    assert scores["Pairwise F-measure"] == 1.0


def test_evaluate_transcription():
    ref = create(
        media_id="audio",
        payload_kind="note_hz",
        extent_kind="time_interval",
        time=[1.0, 2.0],
        duration=[0.5, 0.5],
        value=[440.0, 880.0],
    )
    est = create(
        media_id="audio",
        payload_kind="note_hz",
        extent_kind="time_interval",
        time=[1.01, 2.01],
        duration=[0.49, 0.49],
        value=[440.0, 880.0],
    )

    scores = evaluate(ref, est)
    assert "Precision" in scores
    assert scores["Precision"] == 1.0


def test_evaluate_transcription_midi_notes():
    # 69 is A4 (440 Hz)
    ref = create(
        media_id="audio",
        payload_kind="note_midi",
        extent_kind="time_interval",
        time=[1.0],
        duration=[1.0],
        value=[69.0],
    )
    est = create(
        media_id="audio",
        payload_kind="note_hz",
        extent_kind="time_interval",
        time=[1.01],
        duration=[0.99],
        value=[440.0],
    )

    scores = evaluate(ref, est, task="transcription")
    assert scores["Precision"] == 1.0


def test_evaluate_melody():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ref = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Times(time=[0.0, 0.1, 0.2]),
        payload=PitchContourPayload(
            value=[
                {"frequency": 440.0, "voicing": True},
                {"frequency": 440.0, "voicing": True},
                {"frequency": 0.0, "voicing": False},
            ]
        ),
    )
    est = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Times(time=[0.0, 0.1, 0.2]),
        payload=PitchContourPayload(
            value=[
                {"frequency": 440.0, "voicing": True},
                {"frequency": 440.0, "voicing": True},
                {"frequency": 0.0, "voicing": False},
            ]
        ),
    )
    bopp.validate_and_set_annotation_id(ref)
    bopp.validate_and_set_annotation_id(est)

    scores = evaluate(ref, est)
    assert "Overall Accuracy" in scores
    assert scores["Overall Accuracy"] == 1.0


def test_evaluate_hierarchy():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ref = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=TimeIntervalExtent(time=[0.0, 5.0, 0.0], duration=[5.0, 5.0, 10.0]),
        payload=MultiSegmentPayload(label=["A", "B", "full"], level=[0, 0, 1]),
    )
    est = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=TimeIntervalExtent(time=[0.0, 5.0, 0.0], duration=[5.0, 5.0, 10.0]),
        payload=MultiSegmentPayload(label=["A", "B", "full"], level=[0, 0, 1]),
    )
    bopp.validate_and_set_annotation_id(ref)
    bopp.validate_and_set_annotation_id(est)

    scores = evaluate(ref, est)
    assert "t-measure" in scores


def test_extract_helpers_validation_errors():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")

    # Missing extent for _extract_events
    ann_no_extent = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagOpenPayload(value=["a"]),
    )
    with pytest.raises(BoppArgumentError, match="requires an extent to extract event timestamps"):
        _extract_events(ann_no_extent)

    # Missing extent for _extract_intervals
    with pytest.raises(BoppArgumentError, match="requires an extent to extract intervals"):
        _extract_intervals(ann_no_extent)

    # Missing payload for _extract_labels
    ann_no_payload = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Times(time=[1.0]),
        payload=msgspec.UNSET,  # type: ignore[arg-type]
    )
    with pytest.raises(BoppArgumentError, match="requires a payload to extract labels"):
        _extract_labels(ann_no_payload)

    # Invalid payload for _extract_note_pitches_in_hz
    with pytest.raises(BoppArgumentError, match="Unsupported payload tag"):
        _extract_note_pitches_in_hz(ann_no_extent)


def test_evaluate_unknown_task_and_no_inferred_task():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ann = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagOpenPayload(value=["test"]),
    )
    bopp.validate_and_set_annotation_id(ann)

    with pytest.raises(BoppArgumentError, match="Unsupported evaluation task 'unknown'"):
        evaluate(ann, ann, task="unknown")

    # Annotation with untagged/unknown payload tag
    class UntaggedPayload(msgspec.Struct):
        value: list[int]

    ann_untagged = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=UntaggedPayload(value=[1]),  # type: ignore[arg-type]
    )
    with pytest.raises(BoppArgumentError, match="Could not infer task from reference"):
        evaluate(ann_untagged, ann_untagged)
