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
| jams_val     | ✓           |    20200.2 |     21221.0 |        1266.7 |       40 |
| jams_no_val  | ✗           |      980.9 |      1048.6 |          29.0 |      750 |
| pd_csv       | ✗           |      420.4 |       478.9 |          35.4 |      228 |
| pl_csv       | ✗           |      462.6 |       694.1 |         128.6 |      106 |
| bopp_csv     | ✓           |      846.0 |      1214.8 |         235.4 |       42 |
| bopp_json    | ✓           |      103.8 |       115.6 |           8.6 |     2713 |
| bopp_msgpack | ✓           |       87.5 |        94.2 |           4.9 |     4413 |





Chords
------
Chord estimates of *The Beatles - Drive My Car* as produced by the CREMA model; 87 observed,
disjoint time intervals.

Extents are time intervals in seconds, payloads are chord strings under the Harte grammar,
likelihoods are probabilities (floats in `[0, 1]`).

| Loader       | Validated   |   Min (μs) |   Mean (μs) |   StdDev (μs) |   Rounds |
|:-------------|:------------|-----------:|------------:|--------------:|---------:|
| jams_val     | ✓           |     2166.3 |      2340.5 |         172.9 |      424 |
| jams_no_val  | ✗           |      198.5 |       218.1 |           8.7 |     2935 |
| pd_csv       | ✗           |      413.1 |       455.3 |          24.5 |     1317 |
| pl_csv       | ✗           |      427.8 |       611.7 |         103.3 |     1045 |
| bopp_csv     | ✓           |      713.9 |       896.3 |         137.4 |      334 |
| bopp_json    | ✓           |      115.3 |       131.5 |          12.3 |     4467 |
| bopp_msgpack | ✓           |      107.4 |       118.1 |           6.1 |     3904 |



Legend
------

- `jams_val`: `jams.load(file, validate=True)`
- `jams_no_val`: `jams.load(file, validate=False)`
- `pd_csv`: `pandas.read_csv(file, comment="#")`
- `pl_csv`: `polars.read_csv(file, comment_prefix="#")`
- `bopp_csv`: `bopp.io.load_bopp_csv(file)`
- `bopp_json`: `bopp.io.load_bopp.json(file)`
- `bopp_msgpack`: `bopp.io.load_bopp_msgpack(file)`
