import pandas as pd
import pytest

from bopp.core import create
from bopp.exceptions import BoppValidationError
from bopp.io import (
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.util import _get_tag, from_dataframe


def test_json_roundtrip(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:json_test",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.5, 1.2],
        value=[1, 1, 1],
        sandbox={"user_note": "json_test", "flag": True},
    )
    file_path = tmp_path / "test.json"

    save_bopp_json(ann, file_path)
    loaded = load_bopp_json(file_path)

    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == _get_tag(ann.payload)
    assert getattr(loaded, "sandbox", None) == {"user_note": "json_test", "flag": True}


def test_msgpack_roundtrip(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:msgpack_test",
        payload_kind="onset",
        extent_kind="time",
        time=[0.3, 0.7],
        value=[1, 1],
        sandbox={"user_note": "msgpack_test", "flag": False},
    )
    file_path = tmp_path / "test.msgpack"

    save_bopp_msgpack(ann, file_path)
    loaded = load_bopp_msgpack(file_path)

    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == _get_tag(ann.payload)
    assert getattr(loaded, "sandbox", None) == {"user_note": "msgpack_test", "flag": False}


def test_csv_roundtrip(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:csv_test",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.4],
        value=[1, 1],
        sandbox={"user_note": "csv_test", "count": 10},
    )
    file_path = tmp_path / "test.csv"

    save_bopp_csv(ann, file_path)

    # Test direct loading into Annotation struct
    loaded = load_bopp_csv(file_path)
    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == _get_tag(ann.payload)
    assert getattr(loaded, "sandbox", None) == {"user_note": "csv_test", "count": 10}


def test_csv_column_validation_rules():
    # Rule 1: Column format does not have 3 colon-separated parts
    df_invalid_format = pd.DataFrame({"payload:onset": [1, 2]})
    with pytest.raises(BoppValidationError, match="Invalid column format"):
        from_dataframe(df_invalid_format)

    # Rule 2: Facet has multiple types
    df_mixed_types = pd.DataFrame({
        "extent:time:time": [0.1, 0.2],
        "extent:time_interval:start": [0.1, 0.2],
        "payload:onset:value": [1, 1],
    })
    with pytest.raises(BoppValidationError, match="Mixed types for facet 'extent'"):
        from_dataframe(df_mixed_types)

    # Rule 3: Unrecognized facet or type
    df_invalid_type = pd.DataFrame({
        "payload:nonexistent_type:value": [1, 2]
    })
    with pytest.raises(BoppValidationError, match="Unrecognized type 'nonexistent_type'"):
        from_dataframe(df_invalid_type)

    # Rule 4: Field does not exist on target type struct
    df_invalid_field = pd.DataFrame({
        "payload:onset:invalid_field_name": [1, 2]
    })
    with pytest.raises(BoppValidationError, match="Field 'invalid_field_name' in column .* does not exist"):
        from_dataframe(df_invalid_field)
