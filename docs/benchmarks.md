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
| jams_val     |    14612.1 |     14879.4 |         145.9 |       63 |
| jams_no_val  |      704.5 |       756.8 |          23.2 |     1049 |
| pd_csv       |      287.3 |       334.4 |          36.3 |      305 |
| pl_csv       |      710.0 |      1162.2 |         272.3 |      116 |
| bopp_csv     |      987.7 |      1581.1 |         261.1 |      108 |
| bopp_json    |       77.5 |        91.9 |          10.3 |     4944 |
| bopp_msgpack |       72.6 |        80.1 |          10.2 |     5325 |



Chords
------
Chord estimates of *The Beatles - Drive My Car* as produced by the CREMA model; 87 observed,
disjoint time intervals.

Extents are time intervals in seconds, payloads are chord strings under the Harte grammar,
likelihoods are probabilities (floats in `[0, 1]`).

| Loader       |   Min (μs) |   Mean (μs) |   StdDev (μs) |   Rounds |
|:-------------|-----------:|------------:|--------------:|---------:|
| jams_val     |     1454.8 |      1642.5 |          98.0 |      605 |
| jams_no_val  |      142.9 |       150.9 |           3.9 |     4440 |
| pd_csv       |      311.3 |       355.3 |          31.3 |     1581 |
| pl_csv       |      392.9 |      1251.7 |         258.1 |      804 |
| bopp_csv     |     1063.0 |      1644.4 |         239.2 |      411 |
| bopp_json    |      132.5 |       142.8 |          13.0 |     3316 |
| bopp_msgpack |      147.8 |       160.0 |          14.0 |     2638 |



Legend
------

- `jams_val`: `jams.load(file, validate=True)`
- `jams_no_val`: `jams.load(file, validate=False)`
- `pd_csv`: `pandas.read_csv(file, comment="#")`
- `pl_csv`: `polars.read_csv(file, comment_prefix="#")`
- `bopp_csv`: `bopp.io.load_bopp_csv(file)`
- `bopp_json`: `bopp.io.load_bopp.json(file)`
- `bopp_msgpack`: `bopp.io.load_bopp_msgpack(file)`
