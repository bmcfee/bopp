from __future__ import annotations

import re
from typing import Any

import mir_eval  # type: ignore[import-not-found,import-untyped]
import msgspec
import numpy as np

from .base import BoppBase
from .exceptions import BoppArgumentError
from .util import _get_tag

_ACCIDENTAL_TRANSLATION = str.maketrans(
    {
        "♯": "#",
        "♭": "b",
        "𝄪": "##",
        "𝄫": "bb",
        "♮": "",
    }
)

_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SPN_RE = re.compile(r"^([A-G])([b♭]*|𝄫|[#♯]*|𝄪|♮)?(-?[0-9]+)$")


def _normalize_chord_accidentals(label: str) -> str:
    """Normalize Unicode accidentals to ASCII equivalents for mir_eval compatibility."""
    return label.translate(_ACCIDENTAL_TRANSLATION)


def _spn_to_hz(note_str: str) -> float:
    """Convert a Scientific Pitch Notation string (e.g. 'A4', 'C♯4') to frequency in Hz."""
    m = _SPN_RE.match(note_str)
    if not m:
        raise BoppArgumentError(f"Invalid scientific pitch notation note: '{note_str}'")
    letter, acc, oct_str = m.groups()
    semitone = _PITCH_CLASSES[letter]
    acc_val = 0
    if acc:
        if acc == "𝄪":
            acc_val = 2
        elif acc == "𝄫":
            acc_val = -2
        elif acc == "♮":
            acc_val = 0
        else:
            acc_val += acc.count("#") + acc.count("♯")
            acc_val -= acc.count("b") + acc.count("♭")
    midi_num = (int(oct_str) + 1) * 12 + semitone + acc_val
    return 440.0 * (2.0 ** ((midi_num - 69.0) / 12.0))


def _extract_events(ann: BoppBase) -> Any:
    """Extract 1D event timestamps as a NumPy array."""
    extent = getattr(ann, "extent", None)
    if extent is None or extent is msgspec.UNSET:
        raise BoppArgumentError(
            "Annotation requires an extent to extract event timestamps."
        )

    tag = _get_tag(extent)
    if tag in (
        "time",
        "time_interval",
        "spherical_time",
        "spherical_time_interval",
        "cartesian_time",
        "cartesian_time_interval",
    ):
        return np.asarray(extent.time, dtype=np.float64)
    if tag == "time_uniform":
        payload = getattr(ann, "payload", None)
        n_obs = (
            len(getattr(payload, "value", []))
            if payload not in (None, msgspec.UNSET)
            else 0
        )
        return (
            extent.start_time
            + np.arange(n_obs, dtype=np.float64) * extent.sampling_period
        )

    raise BoppArgumentError(
        f"Cannot extract event timestamps from extent with tag '{tag}'. "
        "Expected temporal extent ('time', 'time_interval', etc.)."
    )


def _extract_intervals(ann: BoppBase) -> Any:
    """Extract (N, 2) interval array [start, end] from an annotation."""
    extent = getattr(ann, "extent", None)
    if extent is None or extent is msgspec.UNSET:
        raise BoppArgumentError("Annotation requires an extent to extract intervals.")

    tag = _get_tag(extent)
    if tag in (
        "time_interval",
        "spherical_time_interval",
        "cartesian_time_interval",
    ):
        starts = np.asarray(extent.time, dtype=np.float64)
        durations = np.asarray(extent.duration, dtype=np.float64)
        if len(starts) == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack([starts, starts + durations])
    if tag in ("time", "spherical_time", "cartesian_time"):
        times = np.asarray(extent.time, dtype=np.float64)
        if len(times) < 2:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack([times[:-1], times[1:]])
    if tag == "time_uniform":
        payload = getattr(ann, "payload", None)
        n_obs = (
            len(getattr(payload, "value", []))
            if payload not in (None, msgspec.UNSET)
            else 0
        )
        if n_obs < 2:
            return np.empty((0, 2), dtype=np.float64)
        times = (
            extent.start_time
            + np.arange(n_obs, dtype=np.float64) * extent.sampling_period
        )
        return np.column_stack([times[:-1], times[1:]])

    raise BoppArgumentError(
        f"Cannot extract intervals from extent with tag '{tag}'. "
        "Expected temporal extent ('time_interval', 'time', etc.)."
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

    if (
        extent_tag in ("time", "spherical_time", "cartesian_time", "time_uniform")
        and len(raw_labels) > 1
    ):
        # Inferred consecutive intervals [t[i], t[i+1]] pair with raw_labels[:-1]
        return list(raw_labels[:-1])

    return list(raw_labels)


def _extract_note_pitches_in_hz(ann: BoppBase) -> Any:
    """Extract note pitches from payload in Hz, converting MIDI note numbers if necessary."""
    payload = getattr(ann, "payload", None)
    payload_tag = _get_tag(payload) if payload not in (None, msgspec.UNSET) else None

    if payload_tag == "frequency_hz":
        return np.asarray(payload.value, dtype=np.float64)  # type: ignore[union-attr]
    if payload_tag == "note_midi":
        midi_nums = np.asarray(payload.value, dtype=np.float64)  # type: ignore[union-attr]
        return 440.0 * (2.0 ** ((midi_nums - 69.0) / 12.0))
    if payload_tag == "note":
        return np.asarray([_spn_to_hz(n) for n in payload.value], dtype=np.float64)  # type: ignore[union-attr]

    raise BoppArgumentError(
        f"Unsupported payload tag '{payload_tag}' for note transcription pitch extraction. "
        "Expected 'frequency_hz', 'note_midi', or 'note'."
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


def _validate_chord_annotation(ann: BoppBase, role: str) -> None:
    """Validate that an annotation satisfies chord evaluation requirements."""
    payload = getattr(ann, "payload", None)
    payload_tag = _get_tag(payload) if payload not in (None, msgspec.UNSET) else None
    if payload_tag != "chord":
        raise BoppArgumentError(
            f"Chord evaluation requires 'chord' payload for {role}, found '{payload_tag}'."
        )

    extent = getattr(ann, "extent", None)
    extent_tag = _get_tag(extent) if extent not in (None, msgspec.UNSET) else None
    if extent_tag not in (
        "time_interval",
        "spherical_time_interval",
        "cartesian_time_interval",
        "time",
        "spherical_time",
        "cartesian_time",
        "time_uniform",
    ):
        raise BoppArgumentError(
            f"Chord evaluation requires temporal extent for {role}, found '{extent_tag}'."
        )

    if not getattr(extent, "disjoint", False):
        raise BoppArgumentError(
            f"Chord evaluation requires extent disjoint=True for {role}, found disjoint={getattr(extent, 'disjoint', False)}."
        )


def _extract_chord_intervals_and_labels(
    ann: BoppBase, role: str
) -> tuple[Any, list[str]]:
    """Extract, sort, and normalize intervals and labels for chord evaluation."""
    _validate_chord_annotation(ann, role)
    raw_intervals = _extract_intervals(ann)
    if len(raw_intervals) == 0:
        raise BoppArgumentError(
            f"Chord evaluation requires non-empty intervals for {role}."
        )
    raw_labels = [
        _normalize_chord_accidentals(lbl) for lbl in _extract_labels(ann, "value")
    ]
    order = np.argsort(raw_intervals[:, 0])
    sorted_intervals = raw_intervals[order]
    sorted_labels = [raw_labels[i] for i in order]
    return sorted_intervals, sorted_labels


def _eval_chord(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_intervals, ref_labels = _extract_chord_intervals_and_labels(ref, "reference")
    est_intervals, est_labels = _extract_chord_intervals_and_labels(est, "estimate")
    return mir_eval.chord.evaluate(
        ref_intervals, ref_labels, est_intervals, est_labels, **kwargs
    )


def _validate_segmentation_annotation(ann: BoppBase, role: str) -> None:
    """Validate that an annotation satisfies segmentation requirements."""
    payload = getattr(ann, "payload", None)
    payload_tag = _get_tag(payload) if payload not in (None, msgspec.UNSET) else None
    if payload_tag != "tag":
        raise BoppArgumentError(
            f"Segmentation evaluation requires 'tag' payload for {role}, found '{payload_tag}'."
        )

    extent = getattr(ann, "extent", None)
    extent_tag = _get_tag(extent) if extent not in (None, msgspec.UNSET) else None
    if extent_tag != "time_interval":
        raise BoppArgumentError(
            f"Segmentation evaluation requires 'time_interval' extent for {role}, found '{extent_tag}'."
        )

    if not getattr(extent, "disjoint", False):
        raise BoppArgumentError(
            f"Segmentation evaluation requires extent disjoint=True for {role}, found disjoint={getattr(extent, 'disjoint', False)}."
        )

    if not getattr(extent, "exhaustive", False):
        raise BoppArgumentError(
            f"Segmentation evaluation requires extent exhaustive=True for {role}, found exhaustive={getattr(extent, 'exhaustive', False)}."
        )


def _extract_segmentation_intervals_and_labels(
    ann: BoppBase, role: str
) -> tuple[Any, list[str]]:
    """Extract and sort intervals and labels for segmentation evaluation."""
    _validate_segmentation_annotation(ann, role)
    extent = ann.extent  # type: ignore[attr-defined]
    payload = ann.payload  # type: ignore[attr-defined]

    starts = np.asarray(extent.time, dtype=np.float64)
    durations = np.asarray(extent.duration, dtype=np.float64)
    labels = [str(x) for x in getattr(payload, "value", [])]

    if len(starts) == 0:
        raise BoppArgumentError(
            f"Segmentation evaluation requires non-empty intervals for {role}."
        )

    order = np.argsort(starts)
    intervals = np.column_stack([starts[order], starts[order] + durations[order]])
    sorted_labels = [labels[i] for i in order]
    return intervals, sorted_labels


def _eval_segment(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_intervals, ref_labels = _extract_segmentation_intervals_and_labels(
        ref, "reference"
    )
    est_intervals, est_labels = _extract_segmentation_intervals_and_labels(
        est, "estimate"
    )
    return mir_eval.segment.evaluate(
        ref_intervals, ref_labels, est_intervals, est_labels, **kwargs
    )


def _extract_multilevel_intervals_and_labels(
    ann: BoppBase, role: str
) -> tuple[list[Any], list[list[str]]]:
    """Extract multi-level intervals and labels for hierarchy evaluation."""
    payload = getattr(ann, "payload", None)
    payload_tag = _get_tag(payload) if payload not in (None, msgspec.UNSET) else None
    if payload_tag != "tag":
        raise BoppArgumentError(
            f"Hierarchy evaluation requires 'tag' payload for {role}, found '{payload_tag}'."
        )

    extent = getattr(ann, "extent", None)
    extent_tag = _get_tag(extent) if extent not in (None, msgspec.UNSET) else None
    if extent_tag != "time_interval":
        raise BoppArgumentError(
            f"Hierarchy evaluation requires 'time_interval' extent for {role}, found '{extent_tag}'."
        )

    groupings = getattr(ann, "groupings", None)
    if groupings is None or groupings is msgspec.UNSET or len(groupings) == 0:
        # Single-level segmentation upcast to hierarchy
        if not getattr(extent, "disjoint", False):
            raise BoppArgumentError(
                f"Hierarchy evaluation without groupings requires extent disjoint=True for {role}, found disjoint=False."
            )
        if not getattr(extent, "exhaustive", False):
            raise BoppArgumentError(
                f"Hierarchy evaluation without groupings requires extent exhaustive=True for {role}, found exhaustive=False."
            )
        intervals, labels = _extract_segmentation_intervals_and_labels(ann, role)
        return [intervals], [labels]

    if len(groupings) > 1:
        raise BoppArgumentError(
            f"Hierarchy evaluation requires at most one grouping on {role} annotation, found {len(groupings)}."
        )

    g = groupings[0]
    if not getattr(g, "disjoint", False):
        raise BoppArgumentError(
            f"Hierarchy evaluation requires grouping disjoint=True for {role}, found disjoint=False on grouping '{g.name}'."
        )
    if not getattr(g, "exhaustive", False):
        raise BoppArgumentError(
            f"Hierarchy evaluation requires grouping exhaustive=True for {role}, found exhaustive=False on grouping '{g.name}'."
        )

    assert extent is not None and extent is not msgspec.UNSET
    raw_starts = np.asarray(extent.time, dtype=np.float64)
    raw_durs = np.asarray(extent.duration, dtype=np.float64)
    raw_labels = [str(x) for x in getattr(payload, "value", [])]

    if len(raw_starts) == 0:
        raise BoppArgumentError(
            f"Hierarchy evaluation requires non-empty intervals for {role}."
        )

    levels = sorted({gid for gid in g.ids if gid >= 0})
    if not levels:
        raise BoppArgumentError(
            f"Hierarchy grouping '{g.name}' on {role} contains no non-negative group IDs."
        )
    if 0 not in levels:
        raise BoppArgumentError(
            f"Hierarchy grouping '{g.name}' on {role} requires root level 0, found levels {levels}."
        )

    multilevel_intervals: list[Any] = []
    multilevel_labels: list[list[str]] = []

    for lvl in levels:
        idx = [i for i, gid in enumerate(g.ids) if gid == lvl]
        lvl_starts = raw_starts[idx]
        lvl_durs = raw_durs[idx]
        lvl_labs = [raw_labels[i] for i in idx]

        order = np.argsort(lvl_starts)
        lvl_intervals = np.column_stack(
            [lvl_starts[order], lvl_starts[order] + lvl_durs[order]]
        )
        lvl_labs = [lvl_labs[j] for j in order]

        multilevel_intervals.append(lvl_intervals)
        multilevel_labels.append(lvl_labs)

    return multilevel_intervals, multilevel_labels


def _eval_hierarchy(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_intervals, ref_labels = _extract_multilevel_intervals_and_labels(
        ref, "reference"
    )
    est_intervals, est_labels = _extract_multilevel_intervals_and_labels(
        est, "estimate"
    )
    return mir_eval.hierarchy.evaluate(
        ref_intervals, ref_labels, est_intervals, est_labels, **kwargs
    )


def _eval_transcription(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_intervals = _extract_intervals(ref)
    est_intervals = _extract_intervals(est)
    ref_pitches = _extract_note_pitches_in_hz(ref)
    est_pitches = _extract_note_pitches_in_hz(est)
    return mir_eval.transcription.evaluate(
        ref_intervals, ref_pitches, est_intervals, est_pitches, **kwargs
    )


def _extract_melody_series(
    ann: BoppBase, role: str, contour: int | None = None
) -> tuple[Any, Any, Any]:
    """
    Extract timestamps, frequencies, and voicing for melody evaluation.

    Parameters
    ----------
    ann : BoppBase
        The annotation to extract from.
    role : str
        The role of the annotation ('reference' or 'estimate').
    contour : int or None, optional
        Target contour ID to extract if groupings are present.
        If None, defaults to 0 if present, else the lowest non-negative group ID.

    Returns
    -------
    times : np.ndarray
        1D array of timestamps sorted chronologically.
    freqs : np.ndarray
        1D array of frequencies in Hz (with unvoiced frames set to 0.0).
    voicing : np.ndarray
        1D array of voicing indicators (1.0 for voiced, 0.0 for unvoiced).
    """
    payload = getattr(ann, "payload", None)
    payload_tag = _get_tag(payload) if payload not in (None, msgspec.UNSET) else None
    if payload_tag != "frequency_hz":
        raise BoppArgumentError(
            f"Melody evaluation requires 'frequency_hz' payload for {role}, found '{payload_tag}'."
        )

    extent = getattr(ann, "extent", None)
    extent_tag = _get_tag(extent) if extent not in (None, msgspec.UNSET) else None
    if extent_tag not in (
        "time",
        "time_uniform",
        "time_interval",
        "spherical_time",
        "spherical_time_interval",
        "cartesian_time",
        "cartesian_time_interval",
    ):
        raise BoppArgumentError(
            f"Melody evaluation requires temporal extent for {role}, found '{extent_tag}'."
        )

    raw_times = _extract_events(ann)
    raw_freqs = np.asarray(getattr(payload, "value", []), dtype=np.float64)
    if len(raw_times) == 0 or len(raw_freqs) == 0:
        raise BoppArgumentError(
            f"Melody evaluation requires non-empty observations for {role}."
        )
    if len(raw_times) != len(raw_freqs):
        raise BoppArgumentError(
            f"Observation times length ({len(raw_times)}) does not match frequencies length ({len(raw_freqs)}) for {role}."
        )

    raw_voiced = getattr(payload, "voiced", None)
    if raw_voiced is None or raw_voiced is msgspec.UNSET:
        voiced_flags = np.ones(len(raw_freqs), dtype=bool)
    else:
        voiced_flags = np.asarray(raw_voiced, dtype=bool)
    if len(voiced_flags) != len(raw_freqs):
        raise BoppArgumentError(
            f"Voicing array length ({len(voiced_flags)}) does not match frequencies length ({len(raw_freqs)}) for {role}."
        )

    groupings = getattr(ann, "groupings", None)
    if groupings is not None and groupings is not msgspec.UNSET and len(groupings) > 0:
        if len(groupings) > 1:
            raise BoppArgumentError(
                f"Melody evaluation requires at most one grouping on {role} annotation, found {len(groupings)}."
            )
        g = groupings[0]
        valid_contours = sorted({gid for gid in g.ids if gid >= 0})
        if not valid_contours:
            raise BoppArgumentError(
                f"Grouping '{g.name}' on {role} contains no non-negative group IDs."
            )
        if contour is None:
            target_contour = 0 if 0 in valid_contours else valid_contours[0]
        else:
            target_contour = contour
            if target_contour not in valid_contours:
                raise BoppArgumentError(
                    f"Contour '{target_contour}' not found in grouping '{g.name}' on {role}; available contours: {valid_contours}."
                )

        idx = [i for i, gid in enumerate(g.ids) if gid == target_contour]
        times = raw_times[idx]
        freqs = raw_freqs[idx]
        voiced = voiced_flags[idx]
        if len(np.unique(times)) != len(times):
            raise BoppArgumentError(
                f"Melody contour '{target_contour}' on {role} contains duplicate timestamps and is not disjoint."
            )
    else:
        if contour is not None and contour != 0:
            raise BoppArgumentError(
                f"Contour '{contour}' requested for {role}, but annotation has no groupings."
            )
        if not getattr(extent, "disjoint", False):
            raise BoppArgumentError(
                f"Melody evaluation without groupings requires extent disjoint=True for {role}, found disjoint=False."
            )
        times = raw_times
        freqs = raw_freqs
        voiced = voiced_flags

    order = np.argsort(times)
    times = times[order]
    freqs = freqs[order]
    voiced = voiced[order]

    effective_voiced = voiced & (freqs > 0)
    freqs_out = np.where(effective_voiced, freqs, 0.0)
    voicing_out = effective_voiced.astype(np.float64)
    return times, freqs_out, voicing_out


def _eval_melody(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_contour = kwargs.pop("ref_contour", kwargs.pop("contour", None))
    est_contour = kwargs.pop("est_contour", kwargs.pop("contour", None))
    ref_time, ref_freq, ref_voicing = _extract_melody_series(
        ref, "reference", contour=ref_contour
    )
    est_time, est_freq, est_voicing = _extract_melody_series(
        est, "estimate", contour=est_contour
    )
    return mir_eval.melody.evaluate(
        ref_time,
        ref_freq,
        est_time,
        est_freq,
        est_voicing=est_voicing,
        ref_reward=ref_voicing,
        **kwargs,
    )


def _extract_multipitch_series(ann: BoppBase, role: str) -> tuple[Any, list[Any]]:
    """
    Extract timestamps and active frequencies list for multipitch evaluation.

    Parameters
    ----------
    ann : BoppBase
        The annotation to extract from.
    role : str
        The role of the annotation ('reference' or 'estimate').

    Returns
    -------
    times : np.ndarray
        1D array of unique timestamps sorted chronologically.
    freqs : list of np.ndarray
        List of 1D arrays of active (voiced, positive) frequencies in Hz at each timestamp.
    """
    payload = getattr(ann, "payload", None)
    payload_tag = _get_tag(payload) if payload not in (None, msgspec.UNSET) else None
    if payload_tag != "frequency_hz":
        raise BoppArgumentError(
            f"Multipitch evaluation requires 'frequency_hz' payload for {role}, found '{payload_tag}'."
        )

    extent = getattr(ann, "extent", None)
    extent_tag = _get_tag(extent) if extent not in (None, msgspec.UNSET) else None
    if extent_tag not in (
        "time",
        "time_uniform",
        "time_interval",
        "spherical_time",
        "spherical_time_interval",
        "cartesian_time",
        "cartesian_time_interval",
    ):
        raise BoppArgumentError(
            f"Multipitch evaluation requires temporal extent for {role}, found '{extent_tag}'."
        )

    raw_times = _extract_events(ann)
    raw_freqs = np.asarray(getattr(payload, "value", []), dtype=np.float64)
    if len(raw_times) == 0 or len(raw_freqs) == 0:
        raise BoppArgumentError(
            f"Multipitch evaluation requires non-empty observations for {role}."
        )
    if len(raw_times) != len(raw_freqs):
        raise BoppArgumentError(
            f"Observation times length ({len(raw_times)}) does not match frequencies length ({len(raw_freqs)}) for {role}."
        )

    raw_voiced = getattr(payload, "voiced", None)
    if raw_voiced is None or raw_voiced is msgspec.UNSET:
        voiced_flags = np.ones(len(raw_freqs), dtype=bool)
    else:
        voiced_flags = np.asarray(raw_voiced, dtype=bool)
    if len(voiced_flags) != len(raw_freqs):
        raise BoppArgumentError(
            f"Voicing array length ({len(voiced_flags)}) does not match frequencies length ({len(raw_freqs)}) for {role}."
        )

    groupings = getattr(ann, "groupings", None)
    if groupings is not None and groupings is not msgspec.UNSET and len(groupings) > 0:
        if len(groupings) > 1:
            raise BoppArgumentError(
                f"Multipitch evaluation requires at most one grouping on {role} annotation, found {len(groupings)}."
            )
        g = groupings[0]
        idx = [i for i, gid in enumerate(g.ids) if gid >= 0]
        if not idx:
            raise BoppArgumentError(
                f"Grouping '{g.name}' on {role} contains no non-negative group IDs."
            )
        times = raw_times[idx]
        freqs = raw_freqs[idx]
        voiced = voiced_flags[idx]
    else:
        times = raw_times
        freqs = raw_freqs
        voiced = voiced_flags

    unique_times = np.unique(times)
    active_mask = voiced & (freqs > 0)

    freqs_list: list[Any] = []
    for t in unique_times:
        t_mask = (times == t) & active_mask
        t_freqs = freqs[t_mask]
        if len(t_freqs) == 0:
            freqs_list.append(np.empty(0, dtype=np.float64))
        else:
            freqs_list.append(np.sort(np.unique(t_freqs)))

    return unique_times, freqs_list


def _eval_multipitch(ref: BoppBase, est: BoppBase, **kwargs: Any) -> dict[str, Any]:
    ref_time, ref_freqs = _extract_multipitch_series(ref, "reference")
    est_time, est_freqs = _extract_multipitch_series(est, "estimate")
    return mir_eval.multipitch.evaluate(
        ref_time, ref_freqs, est_time, est_freqs, **kwargs
    )


# Task dispatch lookup
TASK_DISPATCH = {
    "onset": _eval_onset,
    "beat": _eval_beat,
    "tempo": _eval_tempo,
    "chord": _eval_chord,
    "segment": _eval_segment,
    "segmentation": _eval_segment,
    "transcription": _eval_transcription,
    "hierarchy": _eval_hierarchy,
    "melody": _eval_melody,
    "multipitch": _eval_multipitch,
    "multi_pitch": _eval_multipitch,
}

# Payload tag to task mapping
PAYLOAD_TASK_MAP = {
    "marker": "onset",
    "beat": "beat",
    "tempo": "tempo",
    "chord": "chord",
    "tag": "segment",
    "note_midi": "transcription",
    "note": "transcription",
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
        'segmentation', 'transcription', 'hierarchy', 'melody', 'multipitch').
        If None, the task is inferred from the payload type and extent/grouping attributes of `ref`.
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
    ref_tag = (
        _get_tag(ref_payload) if ref_payload not in (None, msgspec.UNSET) else None
    )

    resolved_task = task
    if resolved_task is None:
        if ref_tag is None:
            raise BoppArgumentError(
                "Could not infer task from reference annotation; no payload tag found."
            )
        if ref_tag == "tag":
            ref_groupings = getattr(ref, "groupings", None)
            if (
                ref_groupings is not None
                and ref_groupings is not msgspec.UNSET
                and len(ref_groupings) == 1
                and getattr(ref_groupings[0], "disjoint", False)
                and getattr(ref_groupings[0], "exhaustive", False)
                and not (
                    getattr(getattr(ref, "extent", None), "disjoint", False)
                    and getattr(getattr(ref, "extent", None), "exhaustive", False)
                )
            ):
                resolved_task = "hierarchy"
            else:
                resolved_task = "segment"
        elif ref_tag == "frequency_hz":
            ref_extent = getattr(ref, "extent", None)
            ref_extent_tag = (
                _get_tag(ref_extent)
                if ref_extent not in (None, msgspec.UNSET)
                else None
            )
            if ref_extent_tag in (
                "time_interval",
                "spherical_time_interval",
                "cartesian_time_interval",
            ):
                resolved_task = "transcription"
            else:
                ref_groupings = getattr(ref, "groupings", None)
                has_multi_contours = (
                    ref_groupings is not None
                    and ref_groupings is not msgspec.UNSET
                    and len(ref_groupings) == 1
                    and len({gid for gid in ref_groupings[0].ids if gid >= 0}) > 1
                )
                is_disjoint = getattr(ref_extent, "disjoint", False)
                if not is_disjoint or has_multi_contours:
                    resolved_task = "multipitch"
                else:
                    resolved_task = "melody"
        else:
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
