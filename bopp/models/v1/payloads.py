# AUTOMATICALLY GENERATED. DO NOT EDIT

from __future__ import annotations

from typing import Annotated

from msgspec import Meta

from .payload import beat, chord

type AnyPayload = Annotated[
    beat.BeatPositionPayload | chord.ChordPayload,
    Meta(
        description="A discriminated union of all supported columnar payload types.",
        title="Any Payload",
    ),
]
