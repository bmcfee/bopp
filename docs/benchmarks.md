Benchmarks
==========

In this section, we'll compare loading and validating annotations between different
serialization formats for BOPP, as well as JAMS (with and without validation) as a baseline
reference.
Comparisons to direct CSV loading via pandas (`pd`) and polars (`pl`) are also included.

All comparisons are implemented on the author's development machine (intel core i7-1360P, 32GB
ram, ubuntu 26.04.01-LTS, python 3.13.0 conda).

Running time is calculated using the `%timeit` utility in IPython.

Beats
-----
1200 observations, synthesized as a 10-minute annotation at 120BPM.

Extents are time values in seconds, payloads are integer-valued.

```
In [3]: %timeit jams.load("longbeats.jams");
32.2 ms ± 1.73 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [4]: %timeit jams.load("longbeats.jams", validate=False);
1.58 ms ± 28.3 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [5]: %timeit pd.read_csv("longbeats.bopp.csv", comment='#')
774 μs ± 15.4 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [6]: %timeit pl.read_csv("longbeats.bopp.csv", comment_prefix="#")
296 μs ± 64.9 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [7]: %timeit bopp.io.load_bopp_csv("longbeats.bopp.csv");
1.64 ms ± 103 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [8]: %timeit bopp.io.load_bopp_json("longbeats.bopp");
180 μs ± 4.01 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [9]: %timeit bopp.io.load_bopp_msgpack("longbeats.bopp.msgpack");
163 μs ± 5.7 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```


Chords
------
Chord estimates of *The Beatles - Drive My Car* as produced by the CREMA model; 87 observed,
disjoint time intervals.

Extents are time intervals in seconds, payloads are chord strings under the Harte grammar,
likelihoods are probabilities (floats in `[0, 1]`).

```
In [3]: %timeit jams.load("drive.jams");
3.68 ms ± 142 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [4]: %timeit jams.load("drive.jams", validate=False);
324 μs ± 9.04 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [5]: %timeit pd.read_csv("drive.csv", comment='#')
861 μs ± 40.2 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [6]: %timeit pl.read_csv("drive.csv", comment_prefix="#")
281 μs ± 63 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [7]: %timeit bopp.io.load_bopp_csv("drive.csv")
2.12 ms ± 65.4 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [8]: %timeit bopp.io.load_bopp_json("drive.bopp")
282 μs ± 9.35 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [9]: %timeit bopp.io.load_bopp_msgpack("drive.bopp.msgpack")
335 μs ± 11.8 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```
