BOPP: The Bounded Observation Payload Protocol
==============================================

When updating the schema, run 

```
hatch run codegen:build
```

## Silly benchmarks

The following is just an initial benchmark comparing `jams.load` and `bopp` deserialization from JSON or msgpack on a representative example file of chord annotations.
Don't take it too seriously, this is just to demonstrate the performance difference between the two libraries.

```
In [14]: %timeit load_bopp_file("drive.bopp")
103 μs ± 2.01 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [13]: %timeit load_from_msgpack("drive.bopp.msgpack")
97.4 μs ± 2.24 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [15]: %timeit jams.load("/home/bmcfee/drive.jams")
3.4 ms ± 106 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [16]: %timeit jams.load("/home/bmcfee/drive.jams", validate=False)
301 μs ± 3.4 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```
