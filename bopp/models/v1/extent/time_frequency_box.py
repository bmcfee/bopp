# AUTOMATICALLY GENERATED. DO NOT EDIT

from __future__ import annotations

from typing import Annotated, Literal

from bopp.base import BoppBase
from msgspec import Meta

type CoordinateItem = Annotated[float, Meta(description="Time", ge=0.0)]


type CoordinateItem1 = Annotated[float, Meta(description="Duration", ge=0.0)]


type CoordinateItem2 = Annotated[
    float, Meta(description="Minimum frequency in Hz", ge=0.0)
]


type CoordinateItem3 = Annotated[
    float, Meta(description="Maximum frequency in Hz", ge=0.0)
]


type Coordinate = Annotated[
    list[CoordinateItem | CoordinateItem1 | CoordinateItem2 | CoordinateItem3],
    Meta(max_length=4, min_length=4),
]


class TimeFrequencyBoxExtent(BoppBase):
    extent_type: Literal["TimeFrequencyBox"]
    coordinates: Annotated[
        list[Coordinate],
        Meta(
            description="An N x 4 array of time-frequency bounding bxoes. Each inner array is [start_time, duration] in seconds."
        ),
    ]
