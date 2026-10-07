from __future__ import annotations

from typing import Any

import mir_eval  # type: ignore[import-not-found,import-untyped]
import msgspec
import numpy as np

from .base import BoppBase
from .exceptions import BoppArgumentError
from .util import _get_tag


def _extract_events(ann: BoppBase) -> Any:
    """Extract 1D event timestamps as a NumPy array."""
    extent = getattr(ann, "extent", None)
    if extent is None or extent is msgspec.UNSET:
        raise BoppArgumentError("Annotation requires an extent to extract event timestamps.")

    tag = _get_tag(extent)
    if tag == "time":
        return np.asarray(extent.time, dtype=np.float64)
    if tag == "time_interval":
        return np.asarray(extent.time, dtype=np.float64)

    raise BoppArgumentError(
        f"Cannot extract event timestamps from extent with tag '{tag}'. "
        "Expected 'time' or 'time_interval'."
    )


def _extract_intervals(ann: BoppBase) -> Any:
    """Extract (N, 2) interval array [start, end] from an annotation."""
    extent = getattr(ann, "extent", None)
    if extent is None or extent is msgspec.UNSET:
        raise BoppArgumentError("Annotation requires an extent to extract intervals.")

    tag = _get_tag(extent)
    if tag == "time_interval":
        starts = np.asarray(extent.time, dtype=np.float64)
        durations = np.asarray(extent.duration, dtype=np.float64)
        if len(starts) == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack([starts, starts + durations])
    if tag == "time":
        times = np.asarray(extent.time, dtype=np.float64)
        if len(times) < 2:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack([times[:-1], times[1:]])

    raise BoppArgumentError(
        f"Cannot extract intervals from extent with tag '{tag}'. "
        "Expected 'time_interval' or 'time'."
    )


def _extract_labels(ann: BoppBase, field_name: str = "value") -> list[Any]:
    """Extract labels list from payload, adapting length if intervals were inferred from point events."""
    payload = getattr(ann, "payload", None)
    if payload is None or payload is msgspec.UNSET:
        raise BoppArgumentError("Annotation requires a payload to extract labels.")

    extent = getattr(ann, "extent", None)
    extent_tag = _get_tag(extent) if extent not in (None, msgspec.UNSET) else None

    raw_labels = getattr(payload, field_name, None)
    if raw_labels is None:
        raise BoppArgumentError(f"Payload has no field '{field_name}'.")

    if extent_tag == "time" and len(raw_labels) > 1:
        # Inferred consecutive intervals [t[i], t[i+1]] pair with raw_labels[:-1]
        return list(raw_labels[:-1])

    return list(raw_labels)


def _extract_note_pitches_in_hz(ann: BoppBase) -> Any:
    """Extract note pitches from payload in Hz, converting MIDI note numbers if necessary."""
    payload = getattr(ann, "payload", None)
    payload_tag = _get_tag(payload) if payload not in (None, msgspec.UNSET) else None

    if payload_tag == "note_hz":
        return np.asarray(payload.value, dtype=np.float64)  # type: ignore[union-attr]
    if payload_tag == "note_midi":
        midi_nums = np.asarray(payload.value, dtype=np.float64)  # type: ignore[union-attr]
        return 440.0 * (2.0 ** ((midi_nums - 69.0) / 12.0))

    raise BoppArgumentError(
        f"Unsupported payload tag '{payload_tag}' for note transcription pitch extraction. "
        "Expected 'note_hz' or 'note_midi'."
    )


def _eval_onset(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_onsets = _extract_events(ref)
    est_onsets = _extract_events(est)
    return mir_eval.onset.evaluate(ref_onsets, est_onsets, **kwargs)


def _eval_beat(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_beats = _extract_events(ref)
    est_beats = _extract_events(est)
    return mir_eval.beat.evaluate(ref_beats, est_beats, **kwargs)


def _eval_tempo(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_payload = getattr(ref, "payload", None)
    est_payload = getattr(est, "payload", None)

    ref_tempi = np.asarray(getattr(ref_payload, "value", []), dtype=np.float64)
    est_tempi = np.asarray(getattr(est_payload, "value", []), dtype=np.float64)

    # mir_eval expects (ref_tempi, ref_weight, est_tempi)
    ref_weight = kwargs.pop("ref_weight", 1.0)
    return mir_eval.tempo.evaluate(ref_tempi, ref_weight, est_tempi, **kwargs)


def _eval_chord(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_intervals = _extract_intervals(ref)
    est_intervals = _extract_intervals(est)
    ref_labels = _extract_labels(ref, "value")
    est_labels = _extract_labels(est, "value")
    return mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels, **kwargs)


def _eval_segment(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_intervals = _extract_intervals(ref)
    est_intervals = _extract_intervals(est)
    ref_labels = _extract_labels(ref, "value")
    est_labels = _extract_labels(est, "value")
    return mir_eval.segment.evaluate(ref_intervals, ref_labels, est_intervals, est_labels, **kwargs)


def _eval_transcription(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_intervals = _extract_intervals(ref)
    est_intervals = _extract_intervals(est)
    ref_pitches = _extract_note_pitches_in_hz(ref)
    est_pitches = _extract_note_pitches_in_hz(est)
    return mir_eval.transcription.evaluate(
        ref_intervals, ref_pitches, est_intervals, est_pitches, **kwargs
    )


def _eval_melody(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    def _extract_times_and_freqs(ann: BoppBase) -> tuple[Any, Any]:
        extent = getattr(ann, "extent", None)
        payload = getattr(ann, "payload", None)

        if extent is not None and extent is not msgspec.UNSET:
            extent_tag = _get_tag(extent)
            if extent_tag == "time":
                times = np.asarray(extent.time, dtype=np.float64)
            else:
                raise BoppArgumentError(
                    f"Melody evaluation requires 'time' extent, found '{extent_tag}'."
                )
        else:
            raise BoppArgumentError("Melody evaluation requires an explicit 'time' extent.")

        raw_vals = getattr(payload, "value", [])
        if raw_vals and isinstance(raw_vals[0], dict):
            # Dict contour entries: {"frequency": float, "voicing": bool/int, ...}
            freqs = []
            for entry in raw_vals:
                f = float(entry.get("frequency", 0.0))
                voicing = entry.get("voicing", True)
                if not voicing or f <= 0.0:
                    freqs.append(-abs(f) if f != 0.0 else 0.0)
                else:
                    freqs.append(f)
            freq_arr = np.asarray(freqs, dtype=np.float64)
        else:
            freq_arr = np.asarray(raw_vals, dtype=np.float64)

        return times, freq_arr

    ref_times, ref_freqs = _extract_times_and_freqs(ref)
    est_times, est_freqs = _extract_times_and_freqs(est)
    return mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs, **kwargs)


def _eval_hierarchy(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    def _extract_multilevel(ann: BoppBase) -> tuple[list[Any], list[list[str]]]:
        extent = getattr(ann, "extent", None)
        payload = getattr(ann, "payload", None)
        extent_tag = _get_tag(extent) if extent not in (None, msgspec.UNSET) else None
        if extent_tag != "time_interval":
            raise BoppArgumentError(
                f"Hierarchy evaluation requires 'time_interval' extent, found '{extent_tag}'."
            )

        times = np.asarray(extent.time, dtype=np.float64)  # type: ignore[union-attr]
        durations = np.asarray(extent.duration, dtype=np.float64)  # type: ignore[union-attr]
        labels = getattr(payload, "label", [])
        levels = getattr(payload, "level", [])

        unique_levels = sorted(set(levels))
        multilevel_intervals: list[Any] = []
        multilevel_labels: list[list[str]] = []

        for lvl in unique_levels:
            idx = [i for i, v in enumerate(levels) if v == lvl]
            lvl_times = times[idx]
            lvl_durs = durations[idx]
            lvl_intervals = np.column_stack([lvl_times, lvl_times + lvl_durs])
            lvl_labels = [str(labels[i]) for i in idx]
            multilevel_intervals.append(lvl_intervals)
            multilevel_labels.append(lvl_labels)

        return multilevel_intervals, multilevel_labels

    ref_intervals, ref_labels = _extract_multilevel(ref)
    est_intervals, est_labels = _extract_multilevel(est)
    return mir_eval.hierarchy.evaluate(
        ref_intervals, ref_labels, est_intervals, est_labels, **kwargs
    )


# Task dispatch lookup
TASK_DISPATCH = {
    "onset": _eval_onset,
    "beat": _eval_beat,
    "tempo": _eval_tempo,
    "chord": _eval_chord,
    "segment": _eval_segment,
    "transcription": _eval_transcription,
    "melody": _eval_melody,
    "hierarchy": _eval_hierarchy,
}

# Payload tag to task mapping
PAYLOAD_TASK_MAP = {
    "onset": "onset",
    "beat": "beat",
    "tempo": "tempo",
    "chord": "chord",
    "segment_open": "segment",
    "tag_open": "segment",
    "note_hz": "transcription",
    "note_midi": "transcription",
    "pitch_contour": "melody",
    "multi_segment": "hierarchy",
}


def evaluate(
    ref: BoppBase,
    est: BoppBase,
    task: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Evaluate an estimated annotation against a reference annotation using mir_eval.

    Parameters
    ----------
    ref : BoppBase
        The ground truth reference annotation.
    est : BoppBase
        The estimated annotation to evaluate.
    task : str or None, optional
        Target evaluation task (e.g. 'onset', 'beat', 'tempo', 'chord', 'segment',
        'transcription', 'melody', 'hierarchy'). If None, the task is inferred from
        the payload type of `ref`.
    **kwargs : Any
        Additional keyword arguments forwarded directly to the corresponding
        `mir_eval.<task>.evaluate` function.

    Returns
    -------
    dict of str to Any
        Evaluation metrics computed by mir_eval.

    Raises
    ------
    BoppArgumentError
        If task cannot be determined or annotations are incompatible.
    """
    ref_payload = getattr(ref, "payload", None)
    est_payload = getattr(est, "payload", None)

    ref_tag = _get_tag(ref_payload) if ref_payload not in (None, msgspec.UNSET) else None
    _get_tag(est_payload) if est_payload not in (None, msgspec.UNSET) else None

    resolved_task = task
    if resolved_task is None:
        if ref_tag is None:
            raise BoppArgumentError(
                "Could not infer task from reference annotation; no payload tag found."
            )
        resolved_task = PAYLOAD_TASK_MAP.get(ref_tag)
        if resolved_task is None:
            raise BoppArgumentError(
                f"No default evaluation task registered for payload tag '{ref_tag}'."
            )

    evaluator = TASK_DISPATCH.get(resolved_task)
    if evaluator is None:
        raise BoppArgumentError(
            f"Unsupported evaluation task '{resolved_task}'. "
            f"Supported tasks: {sorted(TASK_DISPATCH.keys())}"
        )

    return evaluator(ref, est, **kwargs)
