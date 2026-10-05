Benchmarks
==========

In this section, we'll compare loading and validating annotations between different
serialization formats for BOPP, as well as JAMS (with and without validation) as a baseline
reference.
Comparisons to direct CSV loading via pandas (`pd`) and polars (`pl`) are also included.

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
| jams_val     |    28405.8 |     30992.2 |        2860.9 |       26 |
| jams_no_val  |     1376.3 |      1519.1 |         144.0 |      485 |
| pd_csv       |      672.7 |       834.6 |         141.0 |      170 |
| pl_csv       |      320.2 |       538.9 |         105.9 |       20 |
| bopp_csv     |     1455.9 |      1658.4 |         298.4 |       47 |
| bopp_json    |      162.5 |       181.3 |          28.7 |     2330 |
| bopp_msgpack |      140.2 |       158.3 |          22.1 |     2568 |



Chords
------
Chord estimates of *The Beatles - Drive My Car* as produced by the CREMA model; 87 observed,
disjoint time intervals.

Extents are time intervals in seconds, payloads are chord strings under the Harte grammar,
likelihoods are probabilities (floats in `[0, 1]`).

| Loader       |   Min (μs) |   Mean (μs) |   StdDev (μs) |   Rounds |
|:-------------|-----------:|------------:|--------------:|---------:|
| jams_val     |     3024.6 |      3320.7 |         418.9 |      259 |
| jams_no_val  |      275.5 |       300.9 |          24.1 |     2201 |
| pd_csv       |      685.0 |       801.9 |          79.4 |      892 |
| pl_csv       |      156.8 |       274.9 |         149.3 |     1485 |
| bopp_csv     |     1857.5 |      2153.0 |         298.1 |      427 |
| bopp_json    |      255.3 |       294.9 |          35.2 |     1922 |
| bopp_msgpack |      286.0 |       338.9 |          48.7 |     1606 |



Legend
------

- `jams_val`: `jams.load(file, validate=True)`
- `jams_no_val`: `jams.load(file, validate=False)`
- `pd_csv`: `pandas.read_csv(file, comment="#")`
- `pl_csv`: `polars.read_csv(file, comment_prefix="#")`
- `bopp_csv`: `bopp.io.load_bopp_csv(file)`
- `bopp_json`: `bopp.io.load_bopp.json(file)`
- `bopp_msgpack`: `bopp.io.load_bopp_msgpack(file)`
