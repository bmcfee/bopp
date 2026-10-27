# BOPP: The Bounded Observation Payload Protocol

[![CI Tests](https://github.com/bmcfee/bopp/actions/workflows/ci.yml/badge.svg)](https://github.com/bmcfee/bopp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bmcfee/bopp/graph/badge.svg?token=48VE7PS20G)](https://codecov.io/gh/bmcfee/bopp)
[![Documentation Status](https://readthedocs.org/projects/bopp/badge/?version=latest)](http://bopp.readthedocs.io/en/latest/?badge=latest)

## Overview

**BOPP** (Bounded Observation Payload Protocol) is a high-performance, strongly typed annotation framework for audio and music signal processing data built on top of [`msgspec`](https://msgspec.dev/). It provides strict schema validation, fast JSON and MessagePack serialization, and seamless conversion to tabular formats.


### Design

BOPP annotations conform to a flexible but well-defined JSON schema.

The primary contents of a BOPP annotation are the following arrays:
- `payload` an array of observations,
- `extent` (optional) an array of positions (e.g. a time stamp or interval) corresponding to each element of `payload`, and
- `confidence` (optional) an array of confidence ratings (e.g. likelihoods) corresponding to each element of `payload`.

Each of these three fields can take one of a selection of pre-defined types as described in the schema, and their types are identified by the
`payload_type`, `extent_type`, and `confidence_type` fields, as illustrated by the examples below.
All array contents between `payload`, `extent`, and `confidence` are constrained at runtime
(though not in the schema) to have equal length and an implied common index.

All BOPP annotations must contain a `media_id` field to identify the source media being annotated, and a `bopp_version` field to identify the version of the bopp schema used for the annotation.
BOPP annotations may optionally contain structured annotation metadata (e.g. to identify tools or annotators used to produce the contents) and a sandbox field for unstructured data.

### Annotation IDs and Parent Linkage

BOPP annotations include automatic identification and provenance tracking:

- **Annotation ID**: Annotation objects automatically generate a deterministic `annotation_id` based on a UUIDv5 digest computed from the core contents. Any change to the annotation's contents results in a new unique identifier.
- **Parent Linkage**: You can track derivation and lineage across processing pipelines using the `parents` field, which takes a list of parent `annotation_id` strings. Transitive relationships are preserved when deriving annotations through the bopp API (e.g. by trimming or converting between extent types).

```python
>>> import bopp

>>> # Create a parent annotation
>>> parent_ann = bopp.create(
...     media_id="track:123",
...     payload_kind="marker",
...     extent_kind="time",
...     time=[0.1, 0.5, 0.9],
...     description="onset",
... )

>>> bopp.ensure_annotation_id(parent_ann)
'e088e4cb-06d8-591c-b3fb-d11fa81263b5'

>>> child_ann = bopp.transforms.trim(parent_ann, start=0.25)

>>> print(child_ann.parents)
['e088e4cb-06d8-591c-b3fb-d11fa81263b5']

>>> print(child_ann.id)
a9be57d4-7f27-5d84-9e9b-b365112921af
```

### Groupings

A BOPP annotation may optionally contain *groupings*.  A grouping is an integer-valued array of identifiers, equal in length to the observed data (e.g. payload values).
Groupings support a variety of use cases, such as identifying annotation data with audio channels (group ids correspond to left/right), encoding hierarchical annotations (ids
correspond to hierarchical depth), collecting logically related annotation data (e.g. individual pitch contours in an f0 estimation), etc.
Each grouping may optionally include a label mapping to provide string-valued descriptors for each numerical identifier.


### Python implementation

The `bopp` python package provides a reference implementation of the schema and an API for working with annotations.
Note that the schema is the authoritative source for defining a correct annotation; the python implementation is partially generated automatically from the schema by
[`datamodel-code-generator`](https://datamodel-code-generator.koxudaxi.dev/).
The python implementation uses `msgspec.Struct` to implement the annotation objects, though the data can be serialized in a variety of ways (see below).
In princple, BOPP data encoded as json or msgpack should be self-parsing and directly loadable in any programming language which supports those formats.

## Examples

### Global tags

Global annotations (such as track-level tags) contain a payload without requiring an extent array.
This annotation uses the `tag` payload type to indicate that each value is a string with no further constraints (i.e., open vocabulary).

```json
{
    "media_id": "spotify:track:6rqhFgbbK3f2mR3L4A1Lpq",
    "bopp_version": "1.0",
    "payload": {
        "payload_type": "tag",
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

### MIDI intervals

For symbolic MIDI data, extents can be defined in MIDI ticks using `midi_tick` or `midi_interval`. In this example, note pitch values are defined over MIDI tick intervals.

```json
{
  "media_id": "file:midi/song.mid",
  "bopp_version": "1.0",
  "extent": {
    "extent_type": "midi_interval",
    "tick": [0, 480, 960],
    "duration": [480, 480, 960]
  },
  "payload": {
    "payload_type": "note_midi",
    "value": [60, 64, 67]
  }
}
```

### Score position

Sources that store quarter positions as floats can use `score_quarter_float` or `score_interval_float`. These float twins allow negative positions for pickups and nonnegative interval durations.

Musical scores and sheet music annotations can use score-based extents like `score_quarter` (quarter note offsets) or `score_interval`. This example places lyric syllables at musical quarter note positions.

```json
{
  "media_id": "musicxml:score_001",
  "bopp_version": "1.0",
  "extent": {
    "extent_type": "score_quarter",
    "quarter": [[1, 4], [2, 4], [3, 4], [4, 4]]
  },
  "payload": {
    "payload_type": "lyrics",
    "value": ["Hello", "world", "how", "are"],
    "language": "en"
  }
}
```

Convert float score intervals to fractions with an explicit denominator bound, or back to floats:

```python
import bopp

ann = bopp.create(
    media_id="musicxml:score_001",
    payload_kind="tag",
    extent_kind="score_interval_float",
    quarter=[-0.5, 0.0, 1.25],
    duration=[0.5, 1.25, 0.75],
    value=["pickup", "start", "later"],
)
fractions = bopp.to_score_interval(ann, max_denominator=16)
floats = bopp.to_score_interval_float(fractions)
```

Each conversion returns a derived annotation with parent lineage and conversion metadata.
Point extents use `to_score_quarter` and `to_score_quarter_float` in the same way.

### Optical / visual bounding boxes

For visual annotations on sheet music or audio spectrogram images, the `pixel_box` extent defines 2D bounding boxes using `x`, `y`, `width`, and `height` pixel coordinates.

```json
{
  "media_id": "image:page_01.png",
  "bopp_version": "1.0",
  "extent": {
    "extent_type": "pixel_box",
    "x": [120, 250],
    "y": [340, 340],
    "width": [45, 50],
    "height": [60, 60]
  },
  "payload": {
    "payload_type": "object",
    "value": [{"label": "car", "moving": false, "color": "red"},
              {"label": "tree", "moving": false, "color": "green"}]
  }
}
```

### Multimodal extents (spatial, text, and uniform time)

BOPP supports a rich set of 20 standardized extent types spanning multiple observation domains:

- **Temporal**: `time` (continuous timestamps), `time_interval` (timestamps and durations), `time_frequency_box` (time-frequency bounding boxes), and `time_uniform` (implicitly sampled regular grids via `start_time` and `sampling_period`).
- **Spatial & Vision**: `pixel` (2D point coordinates `x, y`), `pixel_box` (2D bounding boxes `x, y, width, height`), `cartesian` (3D coordinates `x, y, z`), and `spherical` (3D coordinates `azimuth, elevation, radius`), with spatiotemporal variants (`cartesian_time`, `cartesian_time_interval`, `spherical_time`, `spherical_time_interval`).
- **Text**: `text` (character point offsets `offset`) and `text_interval` (character spans `offset, length`).
- **Musical Score & Symbolic**: `score_quarter` / `score_quarter_float` (quarter note positions), `score_interval` / `score_interval_float` (quarter note spans), and `midi_tick` / `midi_interval` (MIDI tick positions and durations).

## Dataframe conversion

BOPP annotations can be converted to dataframes using either pandas or polars.
Payload, extent, and confidence data are rendered as columns using a structured naming convention to reflect the types and field names (`facet:type:field`).
For example, the chord annotation given above converts to the following tabular representation:

|   extent:time_interval:time |   extent:time_interval:duration | payload:chord:value   |   confidence:likelihood:confidence |
|----------------------------:|--------------------------------:|:----------------------|-----------------------------------:|
|                         0   |                             1.5 | C:maj                 |                               0.65 |
|                         1.5 |                             3   | F:min7                |                               0.32 |
|                         4.5 |                             1.5 | G:7                   |                               0.77 |

and the beat annotation example given above converts as 

|   extent:time:time |   payload:beat:value |
|-------------------:|---------------------:|
|               0.45 |                    1 |
|               0.98 |                    2 |
|               1.48 |                    3 |
|               2.01 |                    4 |
|               2.5  |                    1 |
|               2.99 |                    2 |
|               3.47 |                    3 |
|               3.98 |                    4 |

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
