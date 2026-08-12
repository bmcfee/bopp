# AUTOMATICALLY GENERATED. DO NOT EDIT

from __future__ import annotations

from typing import Annotated, Literal

from bopp.base import BoppBase
from msgspec import Meta

type Coordinate = Annotated[float, Meta(description="Time", ge=0.0)]


class Timestamps(BoppBase):
    extent_type: Literal["timestamps"]
    coordinates: Annotated[
        list[list[Coordinate]], Meta(description="An N array of time values.")
    ]
