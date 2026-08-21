BOPP: The Bounded Observation Payload Protocol
==============================================

Overview
--------

**BOPP** (Bounded Observation Payload Protocol) is a high-performance, strongly typed annotation framework for audio and music signal processing data built on top of [`msgspec`](https://jcristharif.com/msgspec/). It provides strict schema validation, fast JSON and MessagePack serialization, and seamless conversion to tabular formats (Pandas DataFrames, Apache Arrow, and CSV).

Key Differences from JAMS
-------------------------

While JAMS (JSON Annotated Music Specification) organizes annotations as collections of individual observation dictionaries, BOPP approaches audio annotations with a **columnar/tabular mindset**:

1. **Columnar Data Layout**:
   - *JAMS*: Stores annotations as a list of observation objects, where each observation contains scalar fields (`time`, `duration`, `value`, `confidence`).
   - *BOPP*: Stores annotation facets (`extents`, `payloads`, `confidences`) in parallel, synchronized arrays (e.g., `time` array, `duration` array, `value` array). This layout aligns natively with tabular data structures (DataFrames / Arrow arrays) and enables zero-copy operations and fast vectorized processing.

2. **Strict Typing & Tagged Unions**:
   - *JAMS*: Uses JSON Schema validation over generic dictionaries.
   - *BOPP*: Enforces runtime type-safety through C-accelerated `msgspec.Struct` classes using explicit tagged unions (`payload_type`, `extent_type`, `metadata_type`, `confidence_type`).

3. **Performance & Efficiency**:
   - *JAMS*: Relies on pure-Python dictionary parsing and `jsonschema` validation, which can become a bottleneck during large-scale ML data loading.
   - *BOPP*: Achieves sub-millisecond serialization and validation speeds via C-optimized parsing and efficient MessagePack binary representations.

Example Use Cases
-----------------

- **Machine Learning Data Pipelines**: Rapidly load and batch audio metadata, timestamps, and target payloads (e.g., chords, beats, notes, pitch contours) into training loops without deserialization bottlenecks.
- **Data Analysis & Querying**: Instantly convert annotations to Pandas DataFrames or Apache Arrow tables via `bopp.util.to_dataframe` or `bopp.io.read_bopp_csv` for data manipulation, filtering, and visualization.
- **Interoperability & Interchange**: Export annotations losslessly to CSV or MessagePack for compact storage, sharing, or downstream consumption.

When updating the schema, run 

```
hatch run codegen:build
```

## Silly benchmarks

The following is just an initial benchmark comparing `jams.load` and `bopp` deserialization from JSON or msgpack on a representative example file of chord annotations.
Don't take it too seriously, this is just to demonstrate the performance difference between the two libraries.

```
In [14]: %timeit load_bopp_file("drive.bopp")
103 μs ± 2.01 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [13]: %timeit load_from_msgpack("drive.bopp.msgpack")
97.4 μs ± 2.24 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [15]: %timeit jams.load("/home/bmcfee/drive.jams")
3.4 ms ± 106 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [16]: %timeit jams.load("/home/bmcfee/drive.jams", validate=False)
301 μs ± 3.4 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```
