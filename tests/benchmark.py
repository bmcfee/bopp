import pytest
import jams
import pandas as pd
import polars as pl
import bopp.io

PREFIXES = ["longbeats", "drive"]

LOADERS = {
    "jams_val": lambda p: jams.load(f"tests/bench/{p}.jams"),
    "jams_no_val": lambda p: jams.load(f"tests/bench/{p}.jams", validate=False),
    "pd_csv": lambda p: pd.read_csv(f"tests/bench/{p}.bopp.csv", comment="#"),
    "pl_csv": lambda p: pl.read_csv(f"tests/bench/{p}.bopp.csv", comment_prefix="#"),
    "bopp_csv": lambda p: bopp.io.load_bopp_csv(f"tests/bench/{p}.bopp.csv"),
    "bopp_json": lambda p: bopp.io.load_bopp_json(f"tests/bench/{p}.bopp"),
    "bopp_msgpack": lambda p: bopp.io.load_bopp_msgpack(f"tests/bench/{p}.bopp.msgpack"),
}

@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("loader_name", LOADERS.keys())
def test_io_throughput(benchmark, prefix, loader_name):
    # Benchmark runs the target function dynamically scaled
    benchmark(LOADERS[loader_name], prefix)
