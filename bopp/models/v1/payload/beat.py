# AUTOMATICALLY GENERATED. DO NOT EDIT

from __future__ import annotations

from typing import Annotated

from msgspec import Meta, Struct

type Value = Annotated[int, Meta(ge=1)]


class BeatPositionPayload(Struct, tag_field="payload_type", tag="beat"):
    values: Annotated[
        list[Value],
        Meta(
            description="The metric position of the beat within the measure (e.g., 1 for downbeat, 2, 3, 4)."
        ),
    ]
