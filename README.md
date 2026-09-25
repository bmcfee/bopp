BOPP: The Bounded Observation Payload Protocol
==============================================
[![CI Tests](https://github.com/bmcfee/bopp/actions/workflows/ci.yml/badge.svg)](https://github.com/bmcfee/bopp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bmcfee/bopp/graph/badge.svg?token=48VE7PS20G)](https://codecov.io/gh/bmcfee/bopp)

Overview
--------

**BOPP** (Bounded Observation Payload Protocol) is a high-performance, strongly typed annotation framework for audio and music signal processing data built on top of [`msgspec`](https://jcristharif.com/msgspec/). It provides strict schema validation, fast JSON and MessagePack serialization, and seamless conversion to tabular formats.


Design
------

BOPP annotations conform to a flexible but well-defined JSON schema.  Every BOPP annotation must contain a `media_id` string which identifies the source content that the
annotation describes, the `bopp_version` which identifies the schema version of the annotation.
BOPP annotations may additionally include metadata information (e.g., the tools or annotators responsible for producing the annotation) and a sandbox field for unstructured
additional data.



Examples
--------


