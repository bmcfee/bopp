# AUTOMATICALLY GENERATED. DO NOT EDIT

from __future__ import annotations

from typing import Annotated, Any, Literal

from bopp.base import BoppBase
from msgspec import UNSET, Meta, UnsetType

type ConfidenceItem = Annotated[Any, Meta(ge=0, le=1)]


class LikelihoodConfidence(BoppBase):
    confidence: Annotated[
        list[ConfidenceItem],
        Meta(description="The likeilhood (probability) of each observation"),
    ]
    confidence_type: Literal["likelihood"] | UnsetType = UNSET
