# BOPP: The Bounded Observation Payload Protocol

[![CI Tests](https://github.com/bmcfee/bopp/actions/workflows/ci.yml/badge.svg)](https://github.com/bmcfee/bopp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bmcfee/bopp/graph/badge.svg?token=48VE7PS20G)](https://codecov.io/gh/bmcfee/bopp)

## Overview

**BOPP** (Bounded Observation Payload Protocol) is a high-performance, strongly typed annotation framework for audio and music signal processing data built on top of [`msgspec`](https://jcristharif.com/msgspec/). It provides strict schema validation, fast JSON and MessagePack serialization, and seamless conversion to tabular formats.


## Design

BOPP annotations conform to a flexible but well-defined JSON schema.

The primary contents of a BOPP annotation are the following arrays:
- `payload` an array of observed values
- `extent` (optional) an array of positions (e.g. time) corresponding to each element of `payload`
- `confidence` (optional) an array of confidence ratings (e.g. likelihoods) corresponding to each element of `payload`

Each of these three fields can take one of a selection of pre-defined types as described in the schema and illustrated in the examples below.


## Examples

### Global tags

### Beat and downbeats

### Chord estimations

### Crowd-sourced data


## Serialization

