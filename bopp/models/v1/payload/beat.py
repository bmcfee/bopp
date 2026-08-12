# AUTOMATICALLY GENERATED. DO NOT EDIT

from __future__ import annotations

from typing import Annotated, Literal

from bopp.base import BoppBase
from msgspec import Meta

type ValueItem = Annotated[int, Meta(ge=1)]


class BeatPositionPayload(BoppBase):
    payload_type: Literal["beat"]
    value: Annotated[
        list[ValueItem],
        Meta(
            description="The metric position of the beat within the measure (e.g., 1 for downbeat, 2, 3, 4)."
        ),
    ]
