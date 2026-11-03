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

```{include} benchmarks/longbeats.md
```

Chords
------
Chord estimates of *The Beatles - Drive My Car* as produced by the CREMA model; 87 observed,
disjoint time intervals.

Extents are time intervals in seconds, payloads are chord strings under the Harte grammar,
likelihoods are probabilities (floats in `[0, 1]`).

```{include} benchmarks/drive.md
```

High-Density Vectorized Data (Uniform Time)
-------------------------------------------
100,000 observations of dense frequency measurements (e.g., continuous pitch tracking) sampled at a
uniform period of 10ms (1000 seconds total duration).

Extents are implicitly defined using `time_uniform` (`start_time=0.0`, `sampling_period=0.01`), payloads
are continuous frequencies in Hz (`frequency_hz`), and confidence values represent observation variance
(`variance`). The reference JAMS comparison uses the `pitch_hz` namespace.

```{include} benchmarks/uniform_pitch.md
```

High-Density Vectorized Data (Explicit Time)
--------------------------------------------
The same 100,000 frequency observations as above, but with explicit timestamps serialized via `time`
extent rather than implicit `time_uniform`.

Comparing this against the uniform time benchmark highlights the serialization efficiency and throughput
advantages of implicit time representation for dense, uniformly sampled streams.

```{include} benchmarks/explicit_pitch.md
```

Hierarchical Segmentation
-------------------------
A 3-level hierarchical structural segmentation (parts, sections, phrases; 52 intervals total) for a
10-minute track.

Extents are time intervals (`time_interval`, exhaustive), payloads are string labels (`tag`), and a
grouping layer (`grouping`, named `level`) encodes the structural hierarchy levels. The reference JAMS
comparison uses the `multi_segment` namespace.

```{include} benchmarks/hierarchical.md
```

Lyrics
------
A full song's worth of timed lyrics (155 word-level observations across 6 verses) for the traditional
ballad *The House of the Rising Sun*.

Extents are time intervals (`time_interval`, disjoint) marking each word's onset and duration, and payloads
are word strings under the `lyrics` schema (language tag `en`). The reference JAMS comparison uses the
`lyrics` namespace.

```{include} benchmarks/lyrics.md
```

Legend
------

- `jams_val`: `jams.load(file, validate=True)`
- `jams_no_val`: `jams.load(file, validate=False)`
- `pd_csv`: `pandas.read_csv(file, comment="#")`
- `pl_csv`: `polars.read_csv(file, comment_prefix="#")`
- `bopp_csv`: `bopp.io.load_bopp_csv(file)`
- `bopp_json`: `bopp.io.load_bopp_json(file)`
- `bopp_msgpack`: `bopp.io.load_bopp_msgpack(file)`
