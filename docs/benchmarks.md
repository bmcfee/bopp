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

| Loader       |   Min (μs) |   Mean (μs) |   StdDev (μs) |   Rounds |
|:-------------|-----------:|------------:|--------------:|---------:|
| jams_val     |    13695.6 |     14029.5 |         182.8 |       69 |
| jams_no_val  |      649.2 |       693.5 |          18.6 |     1056 |
| pd_csv       |      279.2 |       305.2 |          27.7 |      283 |
| pl_csv       |      469.1 |      1058.1 |         314.6 |      107 |
| bopp_csv     |     1069.3 |      1570.6 |         223.3 |      126 |
| bopp_json    |       76.3 |        85.4 |          10.4 |     4930 |
| bopp_msgpack |       67.7 |        75.0 |           9.8 |     3881 |



Chords
------
Chord estimates of *The Beatles - Drive My Car* as produced by the CREMA model; 87 observed,
disjoint time intervals.

Extents are time intervals in seconds, payloads are chord strings under the Harte grammar,
likelihoods are probabilities (floats in `[0, 1]`).

| Loader       |   Min (μs) |   Mean (μs) |   StdDev (μs) |   Rounds |
|:-------------|-----------:|------------:|--------------:|---------:|
| jams_val     |     1475.4 |      1551.7 |          85.9 |      641 |
| jams_no_val  |      129.1 |       139.0 |           3.6 |     3840 |
| pd_csv       |      293.7 |       318.6 |          16.1 |     1787 |
| pl_csv       |      574.7 |      1252.5 |         241.3 |      542 |
| bopp_csv     |      926.7 |      1660.4 |         260.6 |      430 |
| bopp_json    |      120.6 |       132.5 |          12.6 |     3492 |
| bopp_msgpack |      137.5 |       148.4 |          13.9 |     2313 |



Legend
------

- `jams_val`: `jams.load(file, validate=True)`
- `jams_no_val`: `jams.load(file, validate=False)`
- `pd_csv`: `pandas.read_csv(file, comment="#")`
- `pl_csv`: `polars.read_csv(file, comment_prefix="#")`
- `bopp_csv`: `bopp.io.load_bopp_csv(file)`
- `bopp_json`: `bopp.io.load_bopp.json(file)`
- `bopp_msgpack`: `bopp.io.load_bopp_msgpack(file)`
