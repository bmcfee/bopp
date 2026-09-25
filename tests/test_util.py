from unittest.mock import patch

import msgspec
import pandas as pd
import polars as pl
import pytest

from bopp.core import BoppArgumentError, BoppIOError, create
from bopp.util import _get_tag, extract_header, from_dataframe, to_dataframe


class TaggedStruct(msgspec.Struct, tag="tagged_sample"):
    name: str


class UntaggedStruct(msgspec.Struct):
    name: str


class NonStruct:
    name: str = "plain_obj"


def test_get_tag():
    assert _get_tag(TaggedStruct(name="a")) == "tagged_sample"
    assert _get_tag(UntaggedStruct(name="b")) is None
    assert _get_tag(NonStruct()) is None


def test_extract_header():
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.2],
        value=[1, 1],
        sandbox={"experiment": "header_test"},
    )
    header = extract_header(ann)
    assert header["media_id"] == "track:1"
    assert header["bopp_version"] == "1.0"
    assert header["sandbox"] == {"experiment": "header_test"}
    assert "payload" not in header
    assert "extent" not in header


def test_to_dataframe_partial_facets():
    # Test annotation without extent or confidence
    ann = create(
        bopp_version="1.0",
        media_id="track:no_extent",
        payload_kind="onset",
        value=[1, 1],
    )
    df = to_dataframe(ann, backend="pandas")
    assert isinstance(df, pd.DataFrame)
    assert df.attrs["media_id"] == "track:no_extent"
    assert not any(col.startswith("extent:") for col in df.columns)
    assert not any(col.startswith("confidence:") for col in df.columns)

    # Reconstitute from DataFrame with missing extent & confidence columns
    reconstructed = from_dataframe(df)
    assert reconstructed.media_id == "track:no_extent"
    assert getattr(reconstructed, "extent", None) is msgspec.UNSET


def test_pandas_dataframe_roundtrip():
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_kind="onset",
        extent_kind="time",
        confidence_kind="likelihood",
        time=[0.1, 0.2],
        value=[1, 1],
        confidence=[0.8, 0.9],
        sandbox={"info": "pandas_test"},
    )

    df = to_dataframe(ann, backend="pandas")
    assert isinstance(df, pd.DataFrame)
    assert df.attrs["media_id"] == "track:1"
    assert df.attrs["sandbox"] == {"info": "pandas_test"}

    reconstructed = from_dataframe(df)
    assert reconstructed.media_id == ann.media_id
    assert _get_tag(reconstructed.payload) == _get_tag(ann.payload)
    assert getattr(reconstructed, "sandbox", None) == {"info": "pandas_test"}


def test_pandas_pyarrow_dataframe_roundtrip():
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_kind="onset",
        extent_kind="time",
        confidence_kind="likelihood",
        time=[0.1, 0.2],
        value=[1, 1],
        confidence=[0.8, 0.9],
        sandbox={"info": "pyarrow_test"},
    )

    df = to_dataframe(ann, backend="pandas-pyarrow")
    assert isinstance(df, pd.DataFrame)
    assert df.attrs["media_id"] == "track:1"
    assert df.attrs["sandbox"] == {"info": "pyarrow_test"}

    reconstructed = from_dataframe(df)
    assert reconstructed.media_id == ann.media_id
    assert _get_tag(reconstructed.payload) == _get_tag(ann.payload)
    assert getattr(reconstructed, "sandbox", None) == {"info": "pyarrow_test"}


def test_polars_dataframe_roundtrip():
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_kind="onset",
        extent_kind="time",
        confidence_kind="likelihood",
        time=[0.1, 0.2],
        value=[1, 1],
        confidence=[0.8, 0.9],
        sandbox={"info": "polars_test"},
    )

    df = to_dataframe(ann, backend="polars")
    assert isinstance(df, pl.DataFrame)
    assert df.attrs["media_id"] == "track:1"
    assert df.attrs["sandbox"] == {"info": "polars_test"}

    reconstructed = from_dataframe(df)
    assert reconstructed.media_id == ann.media_id
    assert _get_tag(reconstructed.payload) == _get_tag(ann.payload)
    assert getattr(reconstructed, "sandbox", None) == {"info": "polars_test"}


def test_from_dataframe_extra_attrs():
    # Test metadata and annotated_domain attributes handling in from_dataframe
    df = pd.DataFrame({"payload:onset:time": [0.1], "payload:onset:value": [1]})
    df.attrs = {
        "bopp_version": "1.0",
        "media_id": "track:extra_attrs",
        "metadata": {"metadata_type": "other", "notes": "nonsense"},
    }

    reconstructed = from_dataframe(df)
    assert reconstructed.media_id == "track:extra_attrs"


def test_missing_backend_imports():
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
    )

    # Test pandas missing
    with patch.dict("sys.modules", {"pandas": None}):
        with pytest.raises(BoppIOError, match="pandas is required"):
            to_dataframe(ann, backend="pandas")

    # Test pyarrow missing
    with patch.dict("sys.modules", {"pyarrow": None}):
        with pytest.raises(BoppIOError, match="pyarrow is required"):
            to_dataframe(ann, backend="pandas-pyarrow")

    # Test polars missing
    with patch.dict("sys.modules", {"polars": None}):
        with pytest.raises(BoppIOError, match="polars is required"):
            to_dataframe(ann, backend="polars")


def test_from_dataframe_missing_pandas_and_polars_imports():
    df = pd.DataFrame({"payload:onset:time": [0.1], "payload:onset:value": [1]})

    # Simulate pandas missing during from_dataframe type check
    with patch.dict("sys.modules", {"pandas": None}):
        with pytest.raises(BoppArgumentError):
            from_dataframe(df)

    # Simulate polars missing during from_dataframe type check
    df_pl = pl.DataFrame({"payload:onset:time": [0.1], "payload:onset:value": [1]})
    with patch.dict("sys.modules", {"polars": None}):
        with pytest.raises(BoppArgumentError):
            from_dataframe(df_pl)


def test_invalid_backend():
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
    )
    with pytest.raises(BoppArgumentError, match="Unsupported backend"):
        to_dataframe(ann, backend="invalid_backend")  # type: ignore[arg-type]


def test_from_dataframe_unsupported_type():
    with pytest.raises(BoppArgumentError, match="Unsupported DataFrame type"):
        from_dataframe({"not": "a_dataframe"})
