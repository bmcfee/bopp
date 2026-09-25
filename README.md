# BOPP: The Bounded Observation Payload Protocol

[![CI Tests](https://github.com/bmcfee/bopp/actions/workflows/ci.yml/badge.svg)](https://github.com/bmcfee/bopp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bmcfee/bopp/graph/badge.svg?token=48VE7PS20G)](https://codecov.io/gh/bmcfee/bopp)

## Overview

**BOPP** (Bounded Observation Payload Protocol) is a high-performance, strongly typed annotation framework for audio and music signal processing data built on top of [`msgspec`](https://jcristharif.com/msgspec/). It provides strict schema validation, fast JSON and MessagePack serialization, and seamless conversion to tabular formats.


### Design

BOPP annotations conform to a flexible but well-defined JSON schema.

The primary contents of a BOPP annotation are the following arrays:
- `payload` an array of observations,
- `extent` (optional) an array of positions (e.g. a time stamp or interval) corresponding to each element of `payload`, and
- `confidence` (optional) an array of confidence ratings (e.g. likelihoods) corresponding to each element of `payload`.

Each of these three fields can take one of a selection of pre-defined types as described in the schema, and their types are identified by the
`payload_type`, `extent_type`, and `confidence_type` fields, as illustrated by the examples below.

All BOPP annotations must contain a `media_id` field to identify the source media being annotated, and a `bopp_version` field to identify the version of the bopp schema used for the annotation.
BOPP annotations may optionally contain structured annotation metadata (e.g. to identify tools or annotators used to produce the contents) and a sandbox field for unstructured data.

### Python implementation

The `bopp` python package provides a reference implementation of the schema and an API for working with annotations.
Note that the schema is the authoritative source for defining a correct annotation; the python implementation is partially generated automatically from the schema by
`datamodel-codegen`.
The python implementation uses `msgspec.Struct` to implement the annotation objects, though the data can be serialized in a variety of ways (see below).
In princple, BOPP data encoded as json or msgpack should be self-parsing and directly loadable in any programming language which supports those formats.

## Examples

### Global tags

Global annotations (such as track-level tags) contain a payload without requiring an extent array.
This annotation uses the `tag_open` payload type to indicate that each value is a string with no further constraints (i.e., open vocabulary).

```json
{
    "media_id": "spotify:track:6rqhFgbbK3f2mR3L4A1Lpq",
    "bopp_version": "1.0",
    "payload": {
        "payload_type": "tag_open",
        "value": ["rock", "classic rock", "70s"]
    }
}
```

### Beat and downbeats

Beat/downbeat annotations are an example where the `time` extent type makes sense.
Each `value` in the payload indicates the bar position (e.g., 1, 2, 3, 4), and the corresponding `time` in the extent indicates when the event happens.
This example also includes optional metadata about how the annotation was produced.

```json
{
  "media_id": "mbid:c8b417c8-04fb-4972-aeaf-161b4742a08d",
  "bopp_version": "1.0",
  "metadata": {
    "metadata_type": "human",
    "annotator_id": "Expert_Alice",
    "tool": "SonicVisualiser"
  },
  "extent": {
    "extent_type": "time",
    "time": [ 0.45, 0.98, 1.48, 2.01, 2.50, 2.99, 3.47, 3.98 ]
  },
  "payload": {
    "payload_type": "beat",
    "value": [1, 2, 3, 4, 1, 2, 3, 4]
  }
}
```

### Chord estimations

Chord annotations logically span a time interval, not an instant.
For this, we can use the `time_interval` extent type which encodes a `time` and `duration` for each observed value in the payload.

This example also uses the `algorithm` metadata type and `likelihood` confidence type, which are both appropriate for outputs
produced by models.

```json
{
  "media_id": "mbid:c8b417c8-04fb-4972-aeaf-161b4742a08d",
  "bopp_version": "1.0",
  "metadata": {
    "metadata_type": "algorithm",
    "algorithm_id": "CREMA",
    "version": "a4c7d57.0",
    "parameters": {}
  }, 
  "extent": {
    "extent_type": "time_interval",
    "time": [0.0, 1.5, 4.5],
    "duration": [1.5, 3.0, 1.5]
  },
  "payload": {
    "payload_type": "chord",
    "value": [ "C:maj", "F:min7", "G:7" ]
  },
  "confidence": {
    "confidence_type": "likelihood",
    "confidence": [0.65, 0.32, 0.77]
  }
}
```

## Dataframe conversion

BOPP annotations can be converted to dataframes using either pandas or polars.
Payload, extent, and confidence data are rendered as columns using a structured naming convention to reflect the types and field names (`facet:type:field`).
For example, the chord annotation given above converts to the following tabular representation:

|    |   extent:time_interval:time |   extent:time_interval:duration | payload:chord:value   |   confidence:likelihood:confidence |
|---:|----------------------------:|--------------------------------:|:----------------------|-----------------------------------:|
|  0 |                         0   |                             1.5 | C:maj                 |                               0.65 |
|  1 |                         1.5 |                             3   | F:min7                |                               0.32 |
|  2 |                         4.5 |                             1.5 | G:7                   |                               0.77 |

Additional information in the BOPP object (`metadata`, `media_id`, etc) are attached as attributes to the dataframe.

## Serialization

While the JSON schema is the ultimate authority for what constitutes a valid BOPP annotation, the data can be serialized to a variety of different formats.
We recommend `msgpack` as the primary format as it is both efficient and does not incur loss of information due to text conversion, but `json` is also supported.
BOPP also provides a structured CSV conversion using dataframes as an intermediate representation, and TOML to encode additional attributes in a structured header.
The chord example above serializes to CSV as follows:

```
# ---
# media_id = "mbid:c8b417c8-04fb-4972-aeaf-161b4742a08d"
# bopp_version = "1.0"
# 
# [metadata]
# metadata_type = "algorithm"
# algorithm_id = "CREMA"
# version = "a4c7d57.0"
# 
# [metadata.parameters]
# ---
extent:time_interval:time,extent:time_interval:duration,payload:chord:value,confidence:likelihood:confidence
0.0,1.5,C:maj,0.65
1.5,3.0,F:min7,0.32
4.5,1.5,G:7,0.77
```
