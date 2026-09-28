Benchmarks
==========

In this section, we'll compare loading and validating annotations between different
serialization formats for BOPP, as well as JAMS (with and without validation) as a baseline
reference.
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

In [5]: %timeit bopp.io.load_bopp_csv("longbeats.bopp.csv");
1.75 ms ± 72.9 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [6]: %timeit bopp.io.load_bopp_json("longbeats.bopp");
252 μs ± 12.2 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [7]: %timeit bopp.io.load_bopp_msgpack("longbeats.bopp.msgpack");
215 μs ± 8.19 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```


Chords
------
Chord estimates of *The Beatles - Drive My Car* as produced by the CREMA model; 87 observed,
disjoint time intervals.

Extents are time intervals in seconds, payloads are chord strings under the Harte grammar,
likelihoods are probabilities (floats in `[0, 1]`).

```
In [8]: %timeit jams.load("drive.jams");
3.68 ms ± 142 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [9]: %timeit jams.load("drive.jams", validate=False);
324 μs ± 9.04 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [10]: %timeit bopp.io.load_bopp_csv("drive.csv")
2.28 ms ± 144 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [11]: %timeit bopp.io.load_bopp_json("drive.bopp")
319 μs ± 11.6 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

In [12]: %timeit bopp.io.load_bopp_msgpack("drive.bopp.msgpack")
314 μs ± 11.2 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```
