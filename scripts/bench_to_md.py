#!/usr/bin/env python
"""Convert benchmark results JSON to markdown tables for documentation."""

import argparse
import json
from pathlib import Path

import pandas as pd


def _format_size(size_bytes: float) -> str:
    """Format file size in human-readable units (B, KB, MB)."""
    if pd.isna(size_bytes):
        return "-"
    size_bytes = float(size_bytes)
    if size_bytes < 1024:
        return f"{int(size_bytes)} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _get_time_unit_and_scale(group: pd.DataFrame) -> tuple[str, float]:
    """
    Determine the most appropriate time unit and scale factor for a benchmark group.

    pytest-benchmark outputs raw time metrics in seconds.
    If the fastest test in the group takes >= 1ms, scale to ms (1e3).
    Otherwise, scale to μs (1e6).
    """
    min_time_sec = group["stats.min"].min()
    if min_time_sec >= 1.0:
        return "s", 1.0
    elif min_time_sec >= 1e-3:
        return "ms", 1e3
    else:
        return "μs", 1e6


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

    # Determine whether the loader includes validation
    df["Validated"] = df["params.loader_name"].apply(
        lambda name: "✓" if (name == "jams_val" or name.startswith("bopp_")) else "✗"
    )

    has_size = "extra_info.file_size" in df.columns
    if has_size:
        df["Size"] = df["extra_info.file_size"].apply(_format_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Group by the parameterized prefix and render separate tables
    for prefix, group in df.groupby("params.prefix"):
        unit, scale = _get_time_unit_and_scale(group)

        group_df = group.copy()
        for col in ["stats.min", "stats.mean", "stats.stddev"]:
            group_df[col] = group_df[col] * scale

        cols = {
            "params.loader_name": "Loader",
            "Validated": "Validated",
        }
        if has_size:
            cols["Size"] = "Size"
        cols.update(
            {
                "stats.min": f"Min ({unit})",
                "stats.mean": f"Mean ({unit})",
                "stats.stddev": f"StdDev ({unit})",
            }
        )

        group_df = group_df[cols.keys()].rename(columns=cols)
        # floatfmt ensures consistent decimal precision via tabulate
        table_md = group_df.to_markdown(index=False, floatfmt=".1f") + "\n"

        out_path = args.output_dir / f"{prefix}.md"
        out_path.write_text(table_md, encoding="utf-8")
        print(f"Wrote benchmark table ({unit}): {out_path}")

        if args.stdout:
            print(f"### Prefix: {prefix} ({unit})\n")
            print(table_md)


if __name__ == "__main__":
    main()
