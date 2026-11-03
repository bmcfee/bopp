from pathlib import Path

import jams
import pandas as pd
import polars as pl
import pytest

import bopp.io
from bopp.util import _get_tag

PREFIXES = [
    "longbeats",
    "drive",
    "uniform_pitch",
    "explicit_pitch",
    "hierarchical",
    "lyrics",
]


def _bopp_to_jams(ann) -> jams.JAMS:
    """Convert a BOPP Annotation to a JAMS object for benchmarking."""
    jam = jams.JAMS()
    payload_tag = _get_tag(ann.payload)
    extent_tag = _get_tag(ann.extent) if ann.extent is not None else None

    # Determine JAMS namespace and values
    has_groupings = bool(getattr(ann, "groupings", None))
    if payload_tag == "frequency_hz":
        namespace = "pitch_hz"
        values = ann.payload.value
    elif payload_tag == "tag" and has_groupings:
        namespace = "multi_segment"
        level_ids = (
            ann.groupings[0].ids.to_pylist()
            if hasattr(ann.groupings[0].ids, "to_pylist")
            else ann.groupings[0].ids
        )
        tag_vals = (
            ann.payload.value.to_pylist()
            if hasattr(ann.payload.value, "to_pylist")
            else ann.payload.value
        )
        values = [
            {"label": str(v), "level": int(lvl)}
            for v, lvl in zip(tag_vals, level_ids)
        ]
    elif payload_tag == "beat":
        namespace = "beat"
        values = ann.payload.value
    elif payload_tag == "chord":
        namespace = "chord"
        values = ann.payload.value
    elif payload_tag == "lyrics":
        namespace = "lyrics"
        values = ann.payload.value
    else:
        namespace = payload_tag
        values = getattr(ann.payload, "value", None)

    if hasattr(values, "to_pylist"):
        values = values.to_pylist()

    jann = jams.Annotation(namespace=namespace)
    n = len(values) if values is not None else 0

    if extent_tag == "time_uniform":
        times = [
            float(ann.extent.start_time + i * ann.extent.sampling_period)
            for i in range(n)
        ]
        durations = [0.0] * n
    elif extent_tag == "time":
        times = (
            ann.extent.time.to_pylist()
            if hasattr(ann.extent.time, "to_pylist")
            else [float(t) for t in ann.extent.time]
        )
        durations = [0.0] * n
    elif extent_tag == "time_interval":
        times = (
            ann.extent.time.to_pylist()
            if hasattr(ann.extent.time, "to_pylist")
            else [float(t) for t in ann.extent.time]
        )
        durations = (
            ann.extent.duration.to_pylist()
            if hasattr(ann.extent.duration, "to_pylist")
            else [float(d) for d in ann.extent.duration]
        )
    else:
        times = [0.0] * n
        durations = [0.0] * n

    confidences = (
        getattr(ann.confidence, "confidence", None)
        if getattr(ann, "confidence", None) is not None
        else None
    )
    if confidences is not None and hasattr(confidences, "to_pylist"):
        confidences = confidences.to_pylist()

    max_t = 0.0
    for i in range(n):
        t = times[i]
        d = durations[i]
        max_t = max(max_t, t + d)
        v = values[i]
        c = (
            float(confidences[i])
            if confidences is not None
            else (1.0 if namespace == "beat" else None)
        )
        jann.append(time=t, duration=d, value=v, confidence=c)

    jam.file_metadata.duration = max(max_t, 1.0)
    jam.annotations.append(jann)
    return jam


@pytest.fixture(scope="module", autouse=True)
def setup_benchmark_data():
    """Generate CSV, msgpack, and JAMS variants from .bopp JSON and clean up afterward."""
    bench_dir = Path(__file__).resolve().parent / "bench"
    created = []
    try:
        for prefix in PREFIXES:
            json_path = bench_dir / f"{prefix}.bopp"
            ann = bopp.io.load_bopp_json(json_path)

            csv_path = bench_dir / f"{prefix}.bopp.csv"
            bopp.io.save_bopp_csv(ann, csv_path)
            created.append(csv_path)

            msgpack_path = bench_dir / f"{prefix}.bopp.msgpack"
            bopp.io.save_bopp_msgpack(ann, msgpack_path)
            created.append(msgpack_path)

            jams_path = bench_dir / f"{prefix}.jams"
            jam = _bopp_to_jams(ann)
            jam.save(str(jams_path))
            created.append(jams_path)

        yield
    finally:
        for path in created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


LOADERS = {
    "jams_val": lambda p: jams.load(f"tests/bench/{p}.jams"),
    "jams_no_val": lambda p: jams.load(f"tests/bench/{p}.jams", validate=False),
    "pd_csv": lambda p: pd.read_csv(f"tests/bench/{p}.bopp.csv", comment="#"),
    "pl_csv": lambda p: pl.read_csv(f"tests/bench/{p}.bopp.csv", comment_prefix="#"),
    "bopp_csv": lambda p: bopp.io.load_bopp_csv(f"tests/bench/{p}.bopp.csv"),
    "bopp_json": lambda p: bopp.io.load_bopp_json(f"tests/bench/{p}.bopp"),
    "bopp_msgpack": lambda p: bopp.io.load_bopp_msgpack(
        f"tests/bench/{p}.bopp.msgpack"
    ),
}

LOADER_FILES = {
    "jams_val": lambda p: Path(f"tests/bench/{p}.jams"),
    "jams_no_val": lambda p: Path(f"tests/bench/{p}.jams"),
    "pd_csv": lambda p: Path(f"tests/bench/{p}.bopp.csv"),
    "pl_csv": lambda p: Path(f"tests/bench/{p}.bopp.csv"),
    "bopp_csv": lambda p: Path(f"tests/bench/{p}.bopp.csv"),
    "bopp_json": lambda p: Path(f"tests/bench/{p}.bopp"),
    "bopp_msgpack": lambda p: Path(f"tests/bench/{p}.bopp.msgpack"),
}


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("loader_name", LOADERS.keys())
def test_io_throughput(benchmark, prefix, loader_name):
    # Record file size in benchmark.extra_info
    file_path = LOADER_FILES[loader_name](prefix)
    benchmark.extra_info["file_size"] = file_path.stat().st_size

    # Benchmark runs the target function dynamically scaled
    benchmark(LOADERS[loader_name], prefix)
