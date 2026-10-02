from unittest.mock import patch

import msgspec
import pandas as pd
import pytest

from bopp.core import create
from bopp.exceptions import BoppArgumentError, BoppValidationError
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


def test_load_bopp_json_invalid_schema(tmp_path):
    # Valid JSON but invalid structure (missing required 'media_id')
    invalid_json_data = '{"bopp_version": "1.0", "payload": {"payload_type": "onset", "time": [0.1], "value": [1]}}'
    file_path = tmp_path / "invalid.json"
    file_path.write_text(invalid_json_data, encoding="utf-8")

    with pytest.raises((msgspec.ValidationError, BoppValidationError)):
        load_bopp_json(file_path)


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


def test_load_bopp_msgpack_invalid_schema(tmp_path):
    # Valid msgpack encoding but missing required fields
    invalid_data = {"media_id": "track:msgpack_test"}
    encoded = msgspec.msgpack.encode(invalid_data)
    file_path = tmp_path / "invalid.msgpack"
    file_path.write_bytes(encoded)

    with pytest.raises((msgspec.ValidationError, BoppValidationError)):
        load_bopp_msgpack(file_path)


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


def test_csv_roundtrip_object_payload(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:csv_obj_test",
        payload_kind="object",
        value=[{"color": "brown"}, {"color": "white"}],
    )
    file_path = tmp_path / "test_obj.csv"

    save_bopp_csv(ann, file_path)

    loaded = load_bopp_csv(file_path)
    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == "object"
    assert getattr(loaded.payload, "value") == [{"color": "brown"}, {"color": "white"}]


def test_load_bopp_csv_string_with_brackets(tmp_path):
    # String tag/payload starting with '[' or '(' should not be parsed via literal_eval
    ann = create(
        bopp_version="1.0",
        media_id="track:bracket_str_test",
        payload_kind="tag_open",
        value=["[intro]", "(chorus)", "{verse}"],
    )
    file_path = tmp_path / "test_bracket_str.csv"

    save_bopp_csv(ann, file_path)

    loaded = load_bopp_csv(file_path)
    assert loaded.media_id == ann.media_id
    assert loaded.payload.value == ["[intro]", "(chorus)", "{verse}"]


def test_load_bopp_csv_invalid_schema(tmp_path):
    # Valid CSV with frontmatter, but columns fail BOPP schema validation (format rule failure)
    csv_content = (
        "# ---\n"
        '# bopp_version = "1.0"\n'
        '# media_id = "track:csv_invalid"\n'
        "# ---\n"
        "payload:onset:nonsense,extent:time:time\n"
        "1,0.1\n"
    )
    file_path = tmp_path / "invalid.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    with pytest.raises(BoppValidationError, match="Field"):
        load_bopp_csv(file_path)


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


def test_save_and_load_with_validate_id_false(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:no_id_val",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
    )

    json_path = tmp_path / "test_no_val.json"
    msgpack_path = tmp_path / "test_no_val.msgpack"
    csv_path = tmp_path / "test_no_val.csv"

    save_bopp_json(ann, json_path, validate_id=False)
    save_bopp_msgpack(ann, msgpack_path, validate_id=False)
    save_bopp_csv(ann, csv_path, validate_id=False)

    loaded_json = load_bopp_json(json_path, validate_id=False)
    loaded_msgpack = load_bopp_msgpack(msgpack_path, validate_id=False)
    loaded_csv = load_bopp_csv(csv_path, validate_id=False)

    assert loaded_json.media_id == ann.media_id
    assert loaded_msgpack.media_id == ann.media_id
    assert loaded_csv.media_id == ann.media_id


def test_save_bopp_csv_unknown_dataframe_type(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:unknown_df",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
    )
    file_path = tmp_path / "unknown_df.csv"

    class MockUnsupportedDataFrame:
        pass

    with patch("bopp.io.to_dataframe", return_value=MockUnsupportedDataFrame()):
        with pytest.raises(BoppArgumentError, match="Unknown dataframe type"):
            save_bopp_csv(ann, file_path)


def test_save_bopp_csv_polars(tmp_path):
    polars = pytest.importorskip("polars")

    ann = create(
        bopp_version="1.0",
        media_id="track:polars_test",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
    )
    file_path = tmp_path / "polars_test.csv"

    with patch("bopp.io.to_dataframe", return_value=polars.DataFrame({"payload:onset:value": [1]})):
        save_bopp_csv(ann, file_path)

    assert file_path.exists()


def test_load_bopp_csv_without_frontmatter(tmp_path):
    csv_content = "payload:onset:value,extent:time:time\n1,0.1\n"
    file_path = tmp_path / "no_frontmatter.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    with patch("bopp.io.from_dataframe") as mock_from_df:
        load_bopp_csv(file_path, validate_id=False)
        mock_from_df.assert_called_once()


def test_load_bopp_csv_non_string_columns(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:numeric_cols",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.2],
        value=[1, 2],
    )
    file_path = tmp_path / "numeric.csv"
    save_bopp_csv(ann, file_path)

    loaded = load_bopp_csv(file_path)
    assert loaded.payload.value == [1, 2]


def test_load_bopp_csv_non_target_columns(tmp_path):
    csv_content = (
        "# ---\n"
        '# bopp_version = "1.0"\n'
        '# media_id = "track:extra_col"\n'
        "# ---\n"
        "payload:onset:value,extent:time:time,custom_col\n"
        "1,0.1,alice\n"
    )
    file_path = tmp_path / "extra_col.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    with patch("bopp.io.from_dataframe") as mock_from_df:
        load_bopp_csv(file_path, validate_id=False)
        mock_from_df.assert_called_once()
