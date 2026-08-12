# AUTOMATICALLY GENERATED. DO NOT EDIT

from __future__ import annotations

from typing import Annotated

from msgspec import Meta

from .extent import time_frequency_box, time_interval, times

type AnyExtent = Annotated[
    times.Timestamps
    | time_interval.TimeIntervalExtent
    | time_frequency_box.TimeFrequencyBoxExtent,
    Meta(
        description="A discriminated union of all supported columnar extent types.",
        title="Any Extent",
    ),
]
