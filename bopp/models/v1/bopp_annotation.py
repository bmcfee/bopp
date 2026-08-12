# AUTOMATICALLY GENERATED. DO NOT EDIT

from __future__ import annotations

from typing import Annotated

from bopp.base import BoppBase
from msgspec import UNSET, Meta, Struct, UnsetType, field

from . import confidences, extents, payloads

type Dependency = Annotated[str, Meta(pattern="^sha256:[a-f0-9]{64}$")]


class Metadata(Struct):
    annotator: str
    tool: str
    version: str


class BoppAnnotation(BoppBase):
    target: Annotated[str, Meta(pattern="^[a-zA-Z0-9]+:.*$")]
    metadata: Metadata
    extent: Annotated[
        extents.AnyExtent,
        Meta(
            description="The parallel array of time/space boundaries (e.g., time_interval, point)"
        ),
    ]
    payload: Annotated[
        payloads.AnyPayload,
        Meta(description="The parallel array of values (e.g., beat, chord)"),
    ]
    dependencies: list[Dependency] | UnsetType = field(default_factory=list)
    confidence: (
        Annotated[
            confidences.AnyConfidence,
            Meta(description="Optional parallel array of likelihoods or votes"),
        ]
        | UnsetType
    ) = UNSET
