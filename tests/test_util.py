import msgspec
import pandas as pd

from bopp.core import create
from bopp.util import _get_tag, extract_header, from_dataframe, to_dataframe


class TaggedStruct(msgspec.Struct, tag="tagged_sample"):
    name: str


class UntaggedStruct(msgspec.Struct):
    name: str


def test_get_tag():
    assert _get_tag(TaggedStruct(name="a")) == "tagged_sample"
    assert _get_tag(UntaggedStruct(name="b")) is None


def test_extract_header():
    ann = create(
        media_id="track:1",
        payload_kind="onset",
        extent_kind="timestamps",
        time=[0.1, 0.2],
        value=[1, 1],
    )
    header = extract_header(ann)
    assert header["media_id"] == "track:1"
    assert header["bopp_version"] == "1.0.0"
    assert "payload" not in header
    assert "extent" not in header


def test_dataframe_roundtrip():
    ann = create(
        media_id="track:1",
        payload_kind="onset",
        extent_kind="timestamps",
        confidence_kind="likelihood",
        time=[0.1, 0.2],
        value=[1, 1],
        confidence=[0.8, 0.9],
    )

    df = to_dataframe(ann)
    assert isinstance(df, pd.DataFrame)
    assert df.attrs["media_id"] == "track:1"

    reconstructed = from_dataframe(df)
    assert reconstructed.media_id == ann.media_id
    assert _get_tag(reconstructed.payload) == _get_tag(ann.payload)
