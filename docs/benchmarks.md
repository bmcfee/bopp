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
| jams_val     |    30910.2 |     45340.5 |       11894.3 |        1 |
| jams_no_val  |     1429.5 |      1676.8 |         190.4 |        1 |
| pd_csv       |      662.7 |       857.1 |         132.1 |        1 |
| pl_csv       |      534.8 |       685.9 |          99.4 |        1 |
| bopp_csv     |     2106.3 |      2938.3 |         983.7 |        1 |
| bopp_json    |      170.7 |       235.7 |          60.5 |        1 |
| bopp_msgpack |      155.9 |       201.9 |          35.6 |        1 |



Chords
------
Chord estimates of *The Beatles - Drive My Car* as produced by the CREMA model; 87 observed,
disjoint time intervals.

Extents are time intervals in seconds, payloads are chord strings under the Harte grammar,
likelihoods are probabilities (floats in `[0, 1]`).

| Loader       |   Min (μs) |   Mean (μs) |   StdDev (μs) |   Rounds |
|:-------------|-----------:|------------:|--------------:|---------:|
| jams_val     |     3158.3 |      3725.9 |         372.8 |        1 |
| jams_no_val  |      295.2 |       398.1 |          54.7 |        1 |
| pd_csv       |      744.9 |      1092.1 |         197.0 |        1 |
| pl_csv       |      156.8 |       267.3 |         161.7 |        1 |
| bopp_csv     |     2009.5 |      2942.4 |         600.3 |        1 |
| bopp_json    |      267.8 |       318.0 |          49.1 |        1 |
| bopp_msgpack |      295.8 |       381.0 |          81.9 |        1 |



Legend
------

- `jams_val`: `jams.load(file, validate=True)`
- `jams_no_val`: `jams.load(file, validate=False)`
- `pd_csv`: `pandas.read_csv(file, comment="#")`
- `pl_csv`: `polars.read_csv(file, comment_prefix="#")`
- `bopp_csv`: `bopp.io.load_bopp_csv(file)`
- `bopp_json`: `bopp.io.load_bopp.json(file)`
- `bopp_msgpack`: `bopp.io.load_bopp_msgpack(file)`
