Benchmarks
==========

In this section, we'll compare loading and validating annotations between different
serialization formats for BOPP, as well as JAMS (with and without validation) as a baseline
reference.
Comparisons to direct CSV loading via pandas (`pd`) and polars (`pl`) are also included.
The BOPP CSV reader uses polars by default with subsequent decoding and schema validation checks.

All comparisons are implemented on the author's development machine (intel core i7-1360P, 32GB
ram, ubuntu 26.04.01-LTS, python 3.13.0 conda).

Running time is calculated by `pytest-benchmark` with code in `tests/benchmark.py` and
configuration in `pyproject.toml`.

Beats
-----
1200 observations, synthesized as a 10-minute annotation at 120BPM.

Extents are time values in seconds, payloads are integer-valued.

| Loader       | Validated   |   Min (μs) |   Mean (μs) |   StdDev (μs) |   Rounds |
|:-------------|:------------|-----------:|------------:|--------------:|---------:|
| jams_val     | ✓           |    13515.3 |     13938.9 |         185.4 |       73 |
| jams_no_val  | ✗           |      626.2 |       678.9 |          20.0 |     1172 |
| pd_csv       | ✗           |      279.0 |       302.1 |          22.9 |      302 |
| pl_csv       | ✗           |      262.4 |       402.4 |         163.9 |      130 |
| bopp_csv     | ✓           |      618.4 |       656.2 |          37.2 |      146 |
| bopp_json    | ✓           |       79.2 |        89.2 |          10.0 |     3643 |
| bopp_msgpack | ✓           |       68.0 |        76.8 |           9.7 |     4927 |




Chords
------
Chord estimates of *The Beatles - Drive My Car* as produced by the CREMA model; 87 observed,
disjoint time intervals.

Extents are time intervals in seconds, payloads are chord strings under the Harte grammar,
likelihoods are probabilities (floats in `[0, 1]`).

| Loader       | Validated   |   Min (μs) |   Mean (μs) |   StdDev (μs) |   Rounds |
|:-------------|:------------|-----------:|------------:|--------------:|---------:|
| jams_val     | ✓           |     1423.3 |      1531.8 |          42.6 |      626 |
| jams_no_val  | ✗           |      127.1 |       136.4 |           5.2 |     4440 |
| pd_csv       | ✗           |      297.7 |       322.6 |          15.9 |     1930 |
| pl_csv       | ✗           |      244.5 |       335.6 |          52.2 |     1548 |
| bopp_csv     | ✓           |      825.0 |       867.8 |          35.0 |     1013 |
| bopp_json    | ✓           |      122.7 |       137.1 |          12.4 |     2850 |
| bopp_msgpack | ✓           |      136.1 |       151.5 |          13.8 |     2460 |



Legend
------

- `jams_val`: `jams.load(file, validate=True)`
- `jams_no_val`: `jams.load(file, validate=False)`
- `pd_csv`: `pandas.read_csv(file, comment="#")`
- `pl_csv`: `polars.read_csv(file, comment_prefix="#")`
- `bopp_csv`: `bopp.io.load_bopp_csv(file)`
- `bopp_json`: `bopp.io.load_bopp.json(file)`
- `bopp_msgpack`: `bopp.io.load_bopp_msgpack(file)`
