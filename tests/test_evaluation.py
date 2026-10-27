import mir_eval
import msgspec
import numpy as np
import pytest

import bopp
from bopp.core import create
from bopp.evaluation import (
    _extract_events,
    _extract_intervals,
    _extract_labels,
    _extract_note_pitches_in_hz,
    _spn_to_hz,
    evaluate,
)
from bopp.exceptions import BoppArgumentError
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.extent.time import Time
from bopp.models.v1.metadata.human import HumanAnnotationMetadata
from bopp.models.v1.payload.tag import TagPayload
from bopp.models.v1.payload.tempo import TempoPayload


def test_evaluate_onset():
    ref_times = [1.0, 2.0, 3.0]
    est_times = [1.02, 2.01, 3.05]

    expected = mir_eval.onset.evaluate(
        np.asarray(ref_times, dtype=float),
        np.asarray(est_times, dtype=float),
    )

    ref = create(
        media_id="audio",
        payload_type="marker",
        extent_type="time",
        time=ref_times,
        description="onset",
    )
    est = create(
        media_id="audio",
        payload_type="marker",
        extent_type="time",
        time=est_times,
        description="onset",
    )

    actual = evaluate(ref, est)
    assert actual == expected
    assert actual["F-measure"] == 1.0


def test_evaluate_onset_from_time_interval():
    ref_times = [1.0, 2.0]
    ref_durs = [0.5, 0.5]
    est_times = [1.01, 2.01]

    expected = mir_eval.onset.evaluate(
        np.asarray(ref_times, dtype=float),
        np.asarray(est_times, dtype=float),
    )

    ref = create(
        media_id="audio",
        payload_type="marker",
        extent_type="time_interval",
        time=ref_times,
        duration=ref_durs,
        description="onset",
    )
    est = create(
        media_id="onset",
        payload_type="marker",
        extent_type="time",
        time=est_times,
        description="onset",
    )
    actual = evaluate(ref, est, task="onset")
    assert actual == expected
    assert actual["F-measure"] == 1.0


def test_evaluate_beat():
    # mir_eval.beat.evaluate trims the first 5 seconds by default (trim=True)
    ref_times = [5.0, 5.5, 6.0, 6.5, 7.0]
    est_times = [5.01, 5.49, 6.02, 6.51, 6.99]

    expected = mir_eval.beat.evaluate(
        np.asarray(ref_times, dtype=float),
        np.asarray(est_times, dtype=float),
    )

    ref = create(
        media_id="audio",
        payload_type="beat",
        extent_type="time",
        time=ref_times,
        value=[1, 2, 3, 4, 1],
    )
    est = create(
        media_id="audio",
        payload_type="beat",
        extent_type="time",
        time=est_times,
        value=[1, 2, 3, 4, 1],
    )

    actual = evaluate(ref, est)
    assert actual == expected
    assert actual["F-measure"] > 0.9


def test_evaluate_tempo():
    ref_tempi = [120.0, 60.0]
    est_tempi = [120.0, 60.0]
    ref_weight = 1.0

    expected = mir_eval.tempo.evaluate(
        np.asarray(ref_tempi, dtype=float),
        ref_weight,
        np.asarray(est_tempi, dtype=float),
    )

    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
    ref = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TempoPayload(value=ref_tempi),
    )
    est = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TempoPayload(value=est_tempi),
    )
    bopp.ensure_annotation_id(ref)
    bopp.ensure_annotation_id(est)

    actual = evaluate(ref, est)
    assert actual == expected
    assert actual["P-score"] == 1.0


def test_evaluate_chord():
    ref_intervals = np.array([[0.0, 2.0], [2.0, 4.0]])
    ref_labels = ["C:maj", "G:maj"]
    est_intervals = np.array([[0.0, 2.0], [2.0, 4.0]])
    est_labels = ["C:maj", "G:maj"]

    expected = mir_eval.chord.evaluate(
        ref_intervals,
        ref_labels,
        est_intervals,
        est_labels,
    )

    ref = create(
        media_id="audio",
        payload_type="chord",
        extent_type="time_interval",
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=ref_labels,
    )
    est = create(
        media_id="audio",
        payload_type="chord",
        extent_type="time_interval",
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=est_labels,
    )

    actual = evaluate(ref, est)
    assert actual == expected
    assert actual["mirex"] == 1.0


def test_evaluate_chord_from_point_times():
    # Consecutive point events infer intervals: [0.0, 2.0] -> 'C:maj', [2.0, 4.0] -> 'G:maj'
    ref = create(
        media_id="audio",
        payload_type="chord",
        extent_type="time",
        time=[0.0, 2.0, 4.0],
        value=["C:maj", "G:maj", "C:maj"],
    )
    est = create(
        media_id="audio",
        payload_type="chord",
        extent_type="time_interval",
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=["C:maj", "G:maj"],
    )

    expected = mir_eval.chord.evaluate(
        np.array([[0.0, 2.0], [2.0, 4.0]]),
        ["C:maj", "G:maj"],
        np.array([[0.0, 2.0], [2.0, 4.0]]),
        ["C:maj", "G:maj"],
    )

    actual = evaluate(ref, est)
    assert actual == expected
    assert actual["mirex"] == 1.0


def test_evaluate_chord_unicode_accidentals():
    # Verify that chords with Unicode accidentals (♯, ♭, 𝄪, 𝄫, ♮) normalize correctly for mir_eval
    ref = create(
        media_id="audio",
        payload_type="chord",
        extent_type="time_interval",
        time=[0.0, 2.0, 4.0],
        duration=[2.0, 2.0, 2.0],
        value=["C♯:maj", "D♭:min", "F𝄪:7"],
    )
    est = create(
        media_id="audio",
        payload_type="chord",
        extent_type="time_interval",
        time=[0.0, 2.0, 4.0],
        duration=[2.0, 2.0, 2.0],
        value=["C#:maj", "Db:min", "F##:7"],
    )

    expected = mir_eval.chord.evaluate(
        np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]]),
        ["C#:maj", "Db:min", "F##:7"],
        np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]]),
        ["C#:maj", "Db:min", "F##:7"],
    )

    actual = evaluate(ref, est)
    assert actual == expected
    assert actual["mirex"] == 1.0


def test_evaluate_segment():
    ref_intervals = np.array([[0.0, 4.0], [4.0, 10.0]])
    ref_labels = ["verse", "chorus"]
    est_intervals = np.array([[0.0, 4.0], [4.0, 10.0]])
    est_labels = ["verse", "chorus"]

    expected = mir_eval.segment.evaluate(
        ref_intervals,
        ref_labels,
        est_intervals,
        est_labels,
    )

    ref = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 4.0],
        duration=[4.0, 6.0],
        value=ref_labels,
    )
    est = create(
        media_id="audio",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 4.0],
        duration=[4.0, 6.0],
        value=est_labels,
    )

    actual = evaluate(ref, est)
    assert actual == expected
    assert actual["Pairwise F-measure"] == 1.0


def test_evaluate_transcription():
    ref_intervals = np.array([[1.0, 1.5], [2.0, 2.5]])
    ref_pitches = np.array([440.0, 880.0])
    est_intervals = np.array([[1.01, 1.5], [2.01, 2.5]])
    est_pitches = np.array([440.0, 880.0])

    expected = mir_eval.transcription.evaluate(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
    )

    ref = create(
        media_id="audio",
        payload_type="frequency_hz",
        extent_type="time_interval",
        time=[1.0, 2.0],
        duration=[0.5, 0.5],
        value=[440.0, 880.0],
    )
    est = create(
        media_id="audio",
        payload_type="frequency_hz",
        extent_type="time_interval",
        time=[1.01, 2.01],
        duration=[0.49, 0.49],
        value=[440.0, 880.0],
    )

    actual = evaluate(ref, est)
    assert actual == expected
    assert actual["Precision"] == 1.0


def test_evaluate_transcription_midi_notes():
    # 69 is A4 (440 Hz)
    ref_intervals = np.array([[1.0, 2.0]])
    ref_pitches = np.array([440.0])
    est_intervals = np.array([[1.01, 2.0]])
    est_pitches = np.array([440.0])

    expected = mir_eval.transcription.evaluate(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
    )

    ref = create(
        media_id="audio",
        payload_type="note_midi",
        extent_type="time_interval",
        time=[1.0],
        duration=[1.0],
        value=[69.0],
    )
    est = create(
        media_id="audio",
        payload_type="frequency_hz",
        extent_type="time_interval",
        time=[1.01],
        duration=[0.99],
        value=[440.0],
    )

    actual = evaluate(ref, est, task="transcription")
    assert actual == expected
    assert actual["Precision"] == 1.0


def test_evaluate_transcription_spn_notes():
    ref_intervals = np.array([[1.0, 2.0]])
    ref_pitches = np.array([440.0])
    est_intervals = np.array([[1.01, 2.0]])
    est_pitches = np.array([440.0])

    expected = mir_eval.transcription.evaluate(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
    )

    ref = create(
        media_id="audio",
        payload_type="note",
        extent_type="time_interval",
        time=[1.0],
        duration=[1.0],
        value=["A4"],
    )
    est = create(
        media_id="audio",
        payload_type="note",
        extent_type="time_interval",
        time=[1.01],
        duration=[0.99],
        value=["A4"],
    )

    actual = evaluate(ref, est)
    assert actual == expected
    assert actual["Precision"] == 1.0


# Note: Melody and hierarchy evaluation tasks are temporarily disabled pending
# reimplementation using group identifiers to encode contour number or hierarchy
# level once grouping schema extensions are implemented.
#
# def test_evaluate_melody():
#     times = np.array([0.0, 0.1, 0.2])
#     ref_freqs = np.array([440.0, 440.0, 0.0])
#     est_freqs = np.array([440.0, 440.0, 0.0])
#
#     expected = mir_eval.melody.evaluate(
#         times,
#         ref_freqs,
#         times,
#         est_freqs,
#     )
#
#     metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
#     ref = Annotation(
#         media_id="audio",
#         bopp_version="1.0.0",
#         metadata=metadata,
#         extent=Time(time=[0.0, 0.1, 0.2]),
#         payload=PitchContourPayload(
#             value=[
#                 {"frequency": 440.0, "voicing": True},
#                 {"frequency": 440.0, "voicing": True},
#                 {"frequency": 0.0, "voicing": False},
#             ]
#         ),
#     )
#     est = Annotation(
#         media_id="audio",
#         bopp_version="1.0.0",
#         metadata=metadata,
#         extent=Time(time=[0.0, 0.1, 0.2]),
#         payload=PitchContourPayload(
#             value=[
#                 {"frequency": 440.0, "voicing": True},
#                 {"frequency": 440.0, "voicing": True},
#                 {"frequency": 0.0, "voicing": False},
#             ]
#         ),
#     )
#     bopp.ensure_annotation_id(ref)
#     bopp.ensure_annotation_id(est)
#
#     actual = evaluate(ref, est)
#     assert actual == expected
#     assert actual["Overall Accuracy"] == 1.0
#
#
# def test_evaluate_hierarchy():
#     ref_intervals = [
#         np.array([[0.0, 5.0], [5.0, 10.0]]),
#         np.array([[0.0, 10.0]]),
#     ]
#     ref_labels = [["A", "B"], ["full"]]
#     est_intervals = [
#         np.array([[0.0, 5.0], [5.0, 10.0]]),
#         np.array([[0.0, 10.0]]),
#     ]
#     est_labels = [["A", "B"], ["full"]]
#
#     with pytest.warns(UserWarning, match="Segment hierarchy is inconsistent"):
#         expected = mir_eval.hierarchy.evaluate(
#             ref_intervals,
#             ref_labels,
#             est_intervals,
#             est_labels,
#         )
#
#     metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")
#     ref = Annotation(
#         media_id="audio",
#         bopp_version="1.0.0",
#         metadata=metadata,
#         extent=TimeIntervalExtent(time=[0.0, 5.0, 0.0], duration=[5.0, 5.0, 10.0]),
#         payload=MultiSegmentPayload(label=["A", "B", "full"], level=[0, 0, 1]),
#     )
#     est = Annotation(
#         media_id="audio",
#         bopp_version="1.0.0",
#         metadata=metadata,
#         extent=TimeIntervalExtent(time=[0.0, 5.0, 0.0], duration=[5.0, 5.0, 10.0]),
#         payload=MultiSegmentPayload(label=["A", "B", "full"], level=[0, 0, 1]),
#     )
#     bopp.ensure_annotation_id(ref)
#     bopp.ensure_annotation_id(est)
#
#     with pytest.warns(UserWarning, match="Segment hierarchy is inconsistent"):
#         actual = evaluate(ref, est)
#     assert actual == expected
#     assert "T-Precision reduced" in actual


def test_extract_helpers_validation_errors():
    metadata = HumanAnnotationMetadata(annotator_id="u1", tool="manual")

    # Missing extent for _extract_events
    ann_no_extent = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=msgspec.UNSET,
        payload=TagPayload(value=["a"]),
    )
    with pytest.raises(
        BoppArgumentError, match="requires an extent to extract event timestamps"
    ):
        _extract_events(ann_no_extent)

    # Missing extent for _extract_intervals
    with pytest.raises(
        BoppArgumentError, match="requires an extent to extract intervals"
    ):
        _extract_intervals(ann_no_extent)

    # Missing payload for _extract_labels
    ann_no_payload = Annotation(
        media_id="audio",
        bopp_version="1.0.0",
        metadata=metadata,
        extent=Time(time=[1.0]),
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
        payload=TagPayload(value=["test"]),
    )
    bopp.ensure_annotation_id(ann)

    with pytest.raises(
        BoppArgumentError, match="Unsupported evaluation task 'unknown'"
    ):
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

    # Valid payload tag but no registered evaluation task (e.g. 'object')
    ann_obj = create(
        media_id="audio",
        payload_type="object",
        extent_type="time",
        time=[0.0],
        value=[{"test": 1}],
    )
    with pytest.raises(
        BoppArgumentError,
        match="No default evaluation task registered for payload tag 'object'",
    ):
        evaluate(ann_obj, ann_obj)


def test_spn_to_hz_accidentals_and_errors():
    assert np.isclose(_spn_to_hz("A4"), 440.0)
    assert np.isclose(_spn_to_hz("C4"), 261.625565)
    assert np.isclose(_spn_to_hz("C♯4"), _spn_to_hz("C#4"))
    assert np.isclose(_spn_to_hz("E♭4"), _spn_to_hz("Eb4"))
    assert np.isclose(_spn_to_hz("D♮4"), _spn_to_hz("D4"))
    assert np.isclose(_spn_to_hz("F𝄪4"), _spn_to_hz("G4"))
    assert np.isclose(_spn_to_hz("B𝄫4"), _spn_to_hz("A4"))

    with pytest.raises(BoppArgumentError, match="Invalid scientific pitch notation"):
        _spn_to_hz("H4")
    with pytest.raises(BoppArgumentError, match="Invalid scientific pitch notation"):
        _spn_to_hz("invalid")
