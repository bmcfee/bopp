from __future__ import annotations

import subprocess
import sys
from typing import Annotated, Literal

import msgspec
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import bopp
from bopp._arrow import (
    DTYPE_STR_TO_PA,
    ArrowArray,
    arrow_dec_hook,
    arrow_json_enc_hook,
    arrow_msgpack_enc_hook,
    coerce_arrow_array,
    unwrap_arrow_array,
)
from bopp.base import ArrowStructBase
from bopp.core import create
from bopp.io import from_dataframe, to_dataframe


class StructF32(ArrowStructBase):
    values: ArrowArray[Literal["float32"], float]


class StructF64(ArrowStructBase):
    values: ArrowArray[Literal["float64"], float]


class StructU8(ArrowStructBase):
    values: ArrowArray[Literal["uint8"], int]


class StructU16(ArrowStructBase):
    values: ArrowArray[Literal["uint16"], int]


class StructU32(ArrowStructBase):
    values: ArrowArray[Literal["uint32"], int]


class StructI32(ArrowStructBase):
    values: ArrowArray[Literal["int32"], int]


class StructBool(ArrowStructBase):
    values: ArrowArray[Literal["bool"], bool]


class StructStr(ArrowStructBase):
    values: ArrowArray[Literal["string"], str]


type ScoreItem = Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]


class ConstrainedFloats(ArrowStructBase):
    scores: ArrowArray[Literal["float32"], ScoreItem]


type CodeItem = Annotated[str, msgspec.Meta(pattern=r"^[A-Z]{3}$")]


class ConstrainedStrings(ArrowStructBase):
    codes: ArrowArray[Literal["string"], CodeItem]


def test_unwrap_arrow_array():
    """Verify unwrap_arrow_array correctly parses ArrowArray type annotations."""
    for dtype_name in DTYPE_STR_TO_PA:
        hint = ArrowArray[Literal[dtype_name], float]
        result = unwrap_arrow_array(hint)
        assert result is not None
        assert result[0] == dtype_name

    # Nested inside Annotated with constraints on item
    constrained_hint = ArrowArray[Literal["float32"], ScoreItem]
    res_constrained = unwrap_arrow_array(constrained_hint)
    assert res_constrained is not None
    assert res_constrained[0] == "float32"

    # Non-arrow types return None
    assert unwrap_arrow_array(list[float]) is None
    assert unwrap_arrow_array(float) is None
    assert unwrap_arrow_array(str) is None
    assert unwrap_arrow_array(None) is None


def test_coerce_arrow_array_containers():
    """Verify coerce_arrow_array handles lists, tuples, numpy, polars, and pyarrow arrays."""
    # List and tuple
    arr_list = coerce_arrow_array([1, 2, 3], "int32")
    assert isinstance(arr_list, pa.Int32Array)
    assert arr_list.to_pylist() == [1, 2, 3]

    arr_tuple = coerce_arrow_array((1.0, 2.0), "float32")
    assert isinstance(arr_tuple, pa.FloatingPointArray)
    assert arr_tuple.type == pa.float32()
    assert arr_tuple.to_pylist() == pytest.approx([1.0, 2.0])

    # Numpy array
    np_arr = np.array([10, 20, 30], dtype=np.uint8)
    arr_np = coerce_arrow_array(np_arr, "uint8")
    assert isinstance(arr_np, pa.UInt8Array)
    assert arr_np.to_pylist() == [10, 20, 30]

    # Polars Series
    pl_series = pl.Series("vals", [100, 200], dtype=pl.UInt16)
    arr_pl = coerce_arrow_array(pl_series, "uint16")
    assert isinstance(arr_pl, pa.UInt16Array)
    assert arr_pl.to_pylist() == [100, 200]

    # Existing pyarrow Array (identity/zero-copy)
    pa_orig = pa.array([True, False, True], type=pa.bool_())
    arr_pa = coerce_arrow_array(pa_orig, "bool")
    assert arr_pa is pa_orig

    # PyArrow ChunkedArray
    chunked = pa.chunked_array([[1, 2], [3, 4]], type=pa.int32())
    arr_chunked = coerce_arrow_array(chunked, "int32")
    assert isinstance(arr_chunked, pa.Int32Array)
    assert arr_chunked.to_pylist() == [1, 2, 3, 4]

    # Unsupported container raises ValidationError
    with pytest.raises(msgspec.ValidationError, match=r"Cannot coerce.*to ArrowArray"):
        coerce_arrow_array({"a": 1}, "int32")


def test_arrow_post_init_coercion():
    """Verify ArrowStructBase.__post_init__ automatically coerces list inputs to pa.Array."""
    obj_f32 = StructF32(values=[0.1, 0.5])
    assert isinstance(obj_f32.values, pa.Array)
    assert obj_f32.values.type == pa.float32()
    assert obj_f32.values.to_pylist() == pytest.approx([0.1, 0.5])

    obj_u8 = StructU8(values=(1, 2))
    assert isinstance(obj_u8.values, pa.Array)
    assert obj_u8.values.type == pa.uint8()
    assert obj_u8.values.to_pylist() == [1, 2]

    obj_str = StructStr(values=np.array(["alpha", "beta"]))
    assert isinstance(obj_str.values, pa.Array)
    assert obj_str.values.type == pa.string()
    assert obj_str.values.to_pylist() == ["alpha", "beta"]

    obj_pl = StructI32(values=pl.Series([10, 20]))
    assert isinstance(obj_pl.values, pa.Array)
    assert obj_pl.values.type == pa.int32()
    assert obj_pl.values.to_pylist() == [10, 20]


@pytest.mark.parametrize(
    "cls, test_values, expected_type",
    [
        (StructF32, [1.0, 2.5, 3.75], pa.float32()),
        (StructF64, [1.0, 2.5, 3.75], pa.float64()),
        (StructU8, [0, 128, 255], pa.uint8()),
        (StructU16, [0, 1000, 65535], pa.uint16()),
        (StructU32, [0, 100000, 4294967295], pa.uint32()),
        (StructI32, [-2147483648, 0, 2147483647], pa.int32()),
        (StructBool, [True, False, True], pa.bool_()),
        (StructStr, ["hello", "world", ""], pa.string()),
    ],
)
def test_all_arrow_dtypes_json_roundtrip(cls, test_values, expected_type):
    """Verify JSON encoding and decoding hooks across all 8 Arrow types."""
    struct = cls(values=test_values)
    assert struct.values.type == expected_type

    # Encode with JSON hook
    json_bytes = msgspec.json.encode(struct, enc_hook=arrow_json_enc_hook)

    # Decode with decode hook
    decoded = msgspec.json.decode(json_bytes, type=cls, dec_hook=arrow_dec_hook)
    assert isinstance(decoded.values, pa.Array)
    assert decoded.values.type == expected_type
    if "float" in str(expected_type):
        assert decoded.values.to_pylist() == pytest.approx(test_values)
    else:
        assert decoded.values.to_pylist() == test_values


@pytest.mark.parametrize(
    "cls, test_values, expected_type",
    [
        (StructF32, [1.5, 2.5, -3.5], pa.float32()),
        (StructF64, [1.5, 2.5, -3.5], pa.float64()),
        (StructU8, [0, 10, 20], pa.uint8()),
        (StructU16, [0, 500, 1000], pa.uint16()),
        (StructU32, [0, 50000, 100000], pa.uint32()),
        (StructI32, [-100, 0, 100], pa.int32()),
        (StructBool, [True, False], pa.bool_()),
        (StructStr, ["abc", "xyz"], pa.string()),
    ],
)
def test_all_arrow_dtypes_msgpack_roundtrip(cls, test_values, expected_type):
    """Verify MsgPack encoding and decoding hooks across all 8 Arrow types."""
    struct = cls(values=test_values)
    assert struct.values.type == expected_type

    # Encode with MsgPack hook
    mp_bytes = msgspec.msgpack.encode(struct, enc_hook=arrow_msgpack_enc_hook)

    # Decode with decode hook
    decoded = msgspec.msgpack.decode(mp_bytes, type=cls, dec_hook=arrow_dec_hook)
    assert isinstance(decoded.values, pa.Array)
    assert decoded.values.type == expected_type
    if "float" in str(expected_type):
        assert decoded.values.to_pylist() == pytest.approx(test_values)
    else:
        assert decoded.values.to_pylist() == test_values


def test_compact_msgpack_raw_float_bytes():
    """Verify float32 and float64 MsgPack serialization formats."""
    # Float32 array
    f32_arr = pa.array([1.0, 2.0], type=pa.float32())
    raw_f32 = arrow_msgpack_enc_hook(f32_arr)
    assert isinstance(raw_f32, msgspec.Raw)
    bytes_f32 = bytes(raw_f32)
    assert bytes_f32[0] == 0x92
    assert bytes_f32[1] == 0xCA

    # Float64 array
    f64_arr = pa.array([1.0, 2.0], type=pa.float64())
    raw_f64 = arrow_msgpack_enc_hook(f64_arr)
    assert isinstance(raw_f64, msgspec.Raw)
    bytes_f64 = bytes(raw_f64)
    assert bytes_f64[0] == 0x92
    assert bytes_f64[1] == 0xCB


def test_arrow_constraint_validation():
    """Verify vectorized constraint validation in arrow_dec_hook."""
    # Valid values decode cleanly
    valid_data = msgspec.json.encode({"scores": [0.0, 0.5, 1.0]})
    decoded = msgspec.json.decode(
        valid_data, type=ConstrainedFloats, dec_hook=arrow_dec_hook
    )
    assert decoded.scores.to_pylist() == pytest.approx([0.0, 0.5, 1.0])

    # Out-of-bounds float < minimum
    invalid_min = msgspec.json.encode({"scores": [-0.1, 0.5]})
    with pytest.raises(msgspec.ValidationError, match="minimum 0.0"):
        msgspec.json.decode(
            invalid_min, type=ConstrainedFloats, dec_hook=arrow_dec_hook
        )

    # Out-of-bounds float > maximum
    invalid_max = msgspec.json.encode({"scores": [0.5, 1.5]})
    with pytest.raises(msgspec.ValidationError, match="maximum 1.0"):
        msgspec.json.decode(
            invalid_max, type=ConstrainedFloats, dec_hook=arrow_dec_hook
        )

    # Valid string pattern
    valid_codes = msgspec.json.encode({"codes": ["ABC", "XYZ"]})
    decoded_codes = msgspec.json.decode(
        valid_codes, type=ConstrainedStrings, dec_hook=arrow_dec_hook
    )
    assert decoded_codes.codes.to_pylist() == ["ABC", "XYZ"]

    # Invalid regex pattern rejection
    invalid_code = msgspec.json.encode({"codes": ["ABC", "123"]})
    with pytest.raises(msgspec.ValidationError, match="does not match pattern"):
        msgspec.json.decode(
            invalid_code, type=ConstrainedStrings, dec_hook=arrow_dec_hook
        )


def test_uint_range_validation():
    """Verify uint bounds checking in arrow_dec_hook."""
    # Negative int rejected
    neg_data = msgspec.json.encode({"values": [-1, 5]})
    with pytest.raises(
        msgspec.ValidationError,
        match=r"(out of range for uint8|can't convert negative value)",
    ):
        msgspec.json.decode(neg_data, type=StructU8, dec_hook=arrow_dec_hook)

    # Over 255 rejected
    overflow_data = msgspec.json.encode({"values": [256]})
    with pytest.raises(
        msgspec.ValidationError,
        match=r"(out of range for uint8|too large to fit|Value.*too large)",
    ):
        msgspec.json.decode(overflow_data, type=StructU8, dec_hook=arrow_dec_hook)


def test_sync_codegen_templates_check_cli():
    """Verify running python scripts/sync_codegen_templates.py --check exits cleanly."""
    try:
        import datamodel_code_generator  # noqa: F401
    except ImportError:
        # Run via hatch in the codegen environment if available
        res = subprocess.run(
            ["hatch", "run", "codegen:python", "scripts/sync_codegen_templates.py", "--check"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "Templates are synchronized" in res.stdout
        return

    result = subprocess.run(
        [sys.executable, "scripts/sync_codegen_templates.py", "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Templates are synchronized" in result.stdout


def test_arrow_annotation_dataframe_roundtrip():
    """Verify DataFrame conversion and roundtrip using native Arrow types."""
    ann = create(
        media_id="track:test",
        payload_type="tag",
        extent_type="time",
        time=[0.5, 1.5, 2.5],
        value=["intro", "verse", "chorus"],
    )

    # Polars DataFrame
    adf_pl = to_dataframe(ann, backend="polars")
    df_pl = adf_pl.df
    assert isinstance(df_pl, pl.DataFrame)
    assert df_pl["extent:time:time"].dtype == pl.Float64
    assert df_pl["payload:tag:value"].dtype == pl.String

    recovered_pl = from_dataframe(df_pl, header=adf_pl.header)
    assert bopp.validate(recovered_pl)
    assert isinstance(recovered_pl.extent.time, pa.Array)
    assert isinstance(recovered_pl.payload.value, pa.Array)
    assert recovered_pl.extent.time.to_pylist() == [0.5, 1.5, 2.5]
    assert recovered_pl.payload.value.to_pylist() == ["intro", "verse", "chorus"]

    # Pandas with PyArrow backend
    adf_pd = to_dataframe(ann, backend="pandas-pyarrow")
    df_pd = adf_pd.df
    assert isinstance(df_pd, pd.DataFrame)
    assert isinstance(df_pd["extent:time:time"].dtype, pd.ArrowDtype)

    recovered_pd = from_dataframe(df_pd, header=adf_pd.header)
    assert bopp.validate(recovered_pd)
    assert isinstance(recovered_pd.extent.time, pa.Array)
    assert isinstance(recovered_pd.payload.value, pa.Array)
    assert recovered_pd.extent.time.to_pylist() == [0.5, 1.5, 2.5]
    assert recovered_pd.payload.value.to_pylist() == ["intro", "verse", "chorus"]
