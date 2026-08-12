# AUTOMATICALLY GENERATED. DO NOT EDIT

from __future__ import annotations

from typing import Annotated, Literal

from bopp.base import BoppBase
from msgspec import Meta

type Coordinate1Item = Annotated[float, Meta(description="Time", ge=0.0)]


type Coordinate1Item1 = Annotated[float, Meta(description="Duration", ge=0.0)]


type Coordinate = Annotated[
    list[Coordinate1Item | Coordinate1Item1], Meta(max_length=2, min_length=2)
]


class TimeIntervalExtent(BoppBase):
    extent_type: Literal["time_interval"]
    coordinates: Annotated[
        list[Coordinate],
        Meta(
            description="An N x 2 array of time intervals. Each inner array is [start_time, duration] in seconds."
        ),
    ]
