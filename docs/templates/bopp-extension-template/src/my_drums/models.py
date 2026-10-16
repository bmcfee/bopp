"""Payload schemas for drum transcription."""

from enum import StrEnum

import msgspec


class DrumComponent(StrEnum):
    """Vocabulary of drum stroke targets."""

    KICK = "kick"
    SNARE = "snare"
    HIHAT_CLOSED = "hihat_closed"
    HIHAT_OPEN = "hihat_open"
    TOM_LOW = "tom_low"
    TOM_HIGH = "tom_high"
    CRASH = "crash"
    RIDE = "ride"


class DrumStroke(msgspec.Struct, forbid_unknown_fields=True):
    """Observation item for a single drum strike."""

    component: DrumComponent
    velocity: int
