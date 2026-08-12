# AUTOMATICALLY GENERATED. DO NOT EDIT

from __future__ import annotations

from typing import Annotated

from msgspec import UNSET, Meta, Struct, UnsetType

type ValueItem = Annotated[
    str,
    Meta(
        pattern="^((N|X)|(([A-G](b*|#*))((:(maj|min|dim|aug|1|5|sus2|sus4|maj6|min6|7|maj7|min7|dim7|hdim7|minmaj7|aug7|9|maj9|min9|11|maj11|min11|13|maj13|min13)(\\((\\*?((b*|#*)([1-9]|1[0-3]?))(,\\*?((b*|#*)([1-9]|1[0-3]?)))*)\\))?)|(:\\((\\*?((b*|#*)([1-9]|1[0-3]?))(,\\*?((b*|#*)([1-9]|1[0-3]?)))*)\\)))?((/((b*|#*)([1-9]|1[0-3]?)))?)?))$"
    ),
]


class ChordPayload(Struct, tag_field="payload_type", tag="chord"):
    value: (
        Annotated[
            list[ValueItem],
            Meta(description="A musical chord string in extended Harte notation."),
        ]
        | UnsetType
    ) = UNSET
