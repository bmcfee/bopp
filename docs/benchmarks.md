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
| jams_val     | ✓           |    14080.6 |     14421.6 |         233.4 |       68 |
| jams_no_val  | ✗           |      650.2 |       703.1 |          22.8 |      942 |
| pd_csv       | ✗           |      281.9 |       316.1 |          58.6 |      273 |
| pl_csv       | ✗           |      340.7 |       594.5 |         248.1 |      108 |
| bopp_csv     | ✓           |      646.2 |       802.4 |          89.3 |       96 |
| bopp_json    | ✓           |       79.5 |        96.0 |          17.6 |     2127 |
| bopp_msgpack | ✓           |       74.5 |        84.2 |          11.9 |     3260 |





Chords
------
Chord estimates of *The Beatles - Drive My Car* as produced by the CREMA model; 87 observed,
disjoint time intervals.

Extents are time intervals in seconds, payloads are chord strings under the Harte grammar,
likelihoods are probabilities (floats in `[0, 1]`).

| Loader       | Validated   |   Min (μs) |   Mean (μs) |   StdDev (μs) |   Rounds |
|:-------------|:------------|-----------:|------------:|--------------:|---------:|
| jams_val     | ✓           |     1486.3 |      1578.5 |          87.5 |      578 |
| jams_no_val  | ✗           |      131.6 |       142.3 |           5.2 |     3802 |
| pd_csv       | ✗           |      300.7 |       352.2 |          27.1 |     2142 |
| pl_csv       | ✗           |      288.1 |       412.8 |          84.7 |     1377 |
| bopp_csv     | ✓           |      771.6 |      1011.3 |         174.5 |      856 |
| bopp_json    | ✓           |      137.3 |       149.5 |          14.6 |     2888 |
| bopp_msgpack | ✓           |      151.4 |       166.6 |          15.6 |     2518 |




Legend
------

- `jams_val`: `jams.load(file, validate=True)`
- `jams_no_val`: `jams.load(file, validate=False)`
- `pd_csv`: `pandas.read_csv(file, comment="#")`
- `pl_csv`: `polars.read_csv(file, comment_prefix="#")`
- `bopp_csv`: `bopp.io.load_bopp_csv(file)`
- `bopp_json`: `bopp.io.load_bopp.json(file)`
- `bopp_msgpack`: `bopp.io.load_bopp_msgpack(file)`
