from pathlib import Path

import jams
import pandas as pd
import polars as pl
import pytest

import bopp.io
from bopp.util import _get_tag

PREFIXES = ["longbeats", "drive"]


def _bopp_to_jams(ann) -> jams.JAMS:
    """Convert a BOPP Annotation to a JAMS object for benchmarking."""
    jam = jams.JAMS()
    namespace = _get_tag(ann.payload)
    jann = jams.Annotation(namespace=namespace)

    times = getattr(ann.extent, "time", None)
    durations = getattr(ann.extent, "duration", None)
    values = getattr(ann.payload, "value", None)
    confidences = (
        getattr(ann.confidence, "confidence", None)
        if ann.confidence is not None
        else None
    )

    n = len(values) if values is not None else (len(times) if times is not None else 0)
    max_t = 0.0
    for i in range(n):
        t = float(times[i]) if times is not None else 0.0
        d = float(durations[i]) if durations is not None else 0.0
        max_t = max(max_t, t + d)
        v = values[i] if values is not None else None
        c = (
            float(confidences[i])
            if confidences is not None
            else (1.0 if namespace == "beat" else None)
        )
        jann.append(time=t, duration=d, value=v, confidence=c)

    jam.file_metadata.duration = max_t
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


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("loader_name", LOADERS.keys())
def test_io_throughput(benchmark, prefix, loader_name):
    # Benchmark runs the target function dynamically scaled
    benchmark(LOADERS[loader_name], prefix)
