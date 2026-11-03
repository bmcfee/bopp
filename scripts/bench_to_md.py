#!/usr/bin/env python
"""Convert benchmark results JSON to markdown tables for documentation."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert benchmark results JSON to markdown tables."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("benchmark.json"),
        help="Path to benchmark JSON results file (default: benchmark.json)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("docs/benchmarks"),
        help="Directory to save individual markdown tables (default: docs/benchmarks)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print rendered tables to stdout in addition to saving files",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data["benchmarks"])

    # pytest-benchmark outputs time metrics in seconds; scale to μs
    time_cols = ["stats.min", "stats.mean", "stats.stddev"]
    df[time_cols] *= 1e6

    # Determine whether the loader includes validation
    df["Validated"] = df["params.loader_name"].apply(
        lambda name: "✓" if (name == "jams_val" or name.startswith("bopp_")) else "✗"
    )

    cols = {
        "params.loader_name": "Loader",
        "Validated": "Validated",
        "stats.min": "Min (μs)",
        "stats.mean": "Mean (μs)",
        "stats.stddev": "StdDev (μs)",
        "stats.rounds": "Rounds",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Group by the parameterized prefix and render separate tables
    for prefix, group in df.groupby("params.prefix"):
        group_df = group[cols.keys()].rename(columns=cols)
        table_md = group_df.to_markdown(index=False, floatfmt=".1f") + "\n"

        out_path = args.output_dir / f"{prefix}.md"
        out_path.write_text(table_md, encoding="utf-8")
        print(f"Wrote benchmark table: {out_path}")

        if args.stdout:
            print(f"### Prefix: {prefix}\n")
            print(table_md)


if __name__ == "__main__":
    main()
