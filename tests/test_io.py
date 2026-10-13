import sys
from fractions import Fraction
from typing import Annotated, Any, Union
from unittest.mock import patch

import msgspec
import pandas as pd
import pytest

from bopp.core import compute_annotation_id, create
from bopp.exceptions import BoppArgumentError, BoppRegistryError, BoppValidationError
from bopp.io import (
    _build_csv_converters,
    _get_column_converter,
    _is_complex_type,
    _prepare_annotation_for_save,
    _unwrap_type,
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


def test_load_bopp_json_missing_id(tmp_path):
    # Valid JSON without an 'id' field
    json_data = '{"bopp_version": "1.0", "media_id": "track:123", "payload": {"payload_type": "onset", "time": [0.1], "value": [1]}}'
    file_path = tmp_path / "missing_id.json"
    file_path.write_text(json_data, encoding="utf-8")

    loaded = load_bopp_json(file_path, validate_id=True)
    assert getattr(loaded, "id", msgspec.UNSET) is msgspec.UNSET


def test_load_bopp_json_mismatched_id_error(tmp_path):
    # Valid UUID string format to pass msgspec schema validation, but computed ID mismatch
    invalid_uuid = "00000000-0000-5000-8000-000000000000"
    json_data = f'{{"bopp_version": "1.0", "media_id": "track:123", "id": "{invalid_uuid}", "payload": {{"payload_type": "onset", "time": [0.1], "value": [1]}}}}'
    file_path = tmp_path / "mismatched_id.json"
    file_path.write_text(json_data, encoding="utf-8")

    with pytest.raises(BoppValidationError, match="Annotation ID mismatch"):
        load_bopp_json(file_path, validate_id=True)


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


def test_csv_roundtrip_numeric_string_tags(tmp_path):
    # Verify string payloads containing pure numbers like "1" or "007" are preserved as strings
    ann = create(
        bopp_version="1.0",
        media_id="track:numeric_str_test",
        payload_kind="tag_open",
        value=["1", "2", "007"],
    )
    file_path = tmp_path / "test_numeric_str.csv"

    save_bopp_csv(ann, file_path)

    loaded = load_bopp_csv(file_path)
    assert loaded.media_id == ann.media_id
    assert loaded.payload.value == ["1", "2", "007"]
    assert all(isinstance(v, str) for v in loaded.payload.value)


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
        generate_id=False,
    )

    json_path = tmp_path / "test_no_val.json"
    msgpack_path = tmp_path / "test_no_val.msgpack"
    csv_path = tmp_path / "test_no_val.csv"

    save_bopp_json(ann, json_path, validate_id=False, generate_id=False)
    save_bopp_msgpack(ann, msgpack_path, validate_id=False, generate_id=False)
    save_bopp_csv(ann, csv_path, validate_id=False, generate_id=False)

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
    # Tests line 497-498: CSV file without '# ---' frontmatter
    csv_content = "payload:onset:value,extent:time:time\n1,0.1\n"
    file_path = tmp_path / "no_frontmatter.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    # Raises BoppValidationError because media_id is missing when frontmatter is omitted
    with pytest.raises(BoppValidationError, match="Missing or invalid required 'media_id'"):
        load_bopp_csv(file_path, validate_id=False)


def test_load_bopp_csv_non_string_columns(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:numeric_cols",
        payload_kind="beat",
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
        "payload:onset:value,extent:time:time\n"
        "1,0.1\n"
    )
    file_path = tmp_path / "extra_col.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    loaded = load_bopp_csv(file_path, validate_id=True)
    assert loaded.media_id == "track:extra_col"


# ==========================================
# Unit tests targeting previously missing coverage in bopp/io.py
# ==========================================


def test_unwrap_type_pep695_and_multi_union():
    # Test lines 63-72: TypeAliasType and multi-type unions
    if sys.version_info >= (3, 12):
        namespace: dict[str, Any] = {}
        exec("type MyAlias = int", namespace)
        assert _unwrap_type(namespace["MyAlias"]) is int

    # Multi-type union where unwrap stops immediately
    multi_union = Union[int, str]
    assert _unwrap_type(multi_union) == multi_union

    # Multi-type union with None (covers line 70 else: break, returns original union unchanged)
    multi_union_with_none = Union[int, str, None]
    assert _unwrap_type(multi_union_with_none) == multi_union_with_none

    # Annotated wrapping an alias/union
    annotated_type = Annotated[int, "metadata"]
    assert _unwrap_type(annotated_type) is int


def test_is_complex_type_branches():
    # Test lines 107-108, 114-118
    # 1. Union containing a complex type
    union_with_complex = Union[list[int], int]
    assert _is_complex_type(union_with_complex) is True

    # 2. Custom class that is not primitive nor struct/collection
    class DummyCustomClass:
        pass

    assert _is_complex_type(DummyCustomClass) is False

    # 3. String forward reference containing complex identifiers
    assert _is_complex_type("Fraction") is True
    assert _is_complex_type("list[str]") is True
    assert _is_complex_type("int") is False


def test_column_converters_edge_cases():
    # Test lines 146->152, 148->152, 156, 169
    converter_complex = _get_column_converter(list[list[int]])
    assert converter_complex is not None
    # None on non-string or whitespace
    assert converter_complex(None) is None  # type: ignore[arg-type]
    assert converter_complex("   ") is None
    # Branches 146->152 and 148->152: evaluating valid container literals
    assert converter_complex("[1, 2, 3]") == [1, 2, 3]
    assert converter_complex("(4, 5)") == (4, 5)
    assert converter_complex('{"key": "val"}') == {"key": "val"}
    # Literal not starting with [, (, { returns raw string
    assert converter_complex("unbracketed_val") == "unbracketed_val"

    # String converter handling empty string
    converter_str = _get_column_converter(list[str])
    assert converter_str is not None
    assert converter_str("") == ""
    assert converter_str("abc") == "abc"


def test_build_csv_converters_error_and_edge_cases():
    # Test lines 195, 202-203, 208->206, 211->206, 214->206, 217-218
    # 1. Empty header line
    assert _build_csv_converters("", "1.0") == {}

    # 2. Unknown bopp_version triggering BoppRegistryError
    assert _build_csv_converters("payload:onset:value", "nonexistent_version") == {}

    # 3. Malformed column / non-facet column / unrecognized tag / missing field
    header = (
        "not_three_parts,"
        "unknown_facet:foo:bar,"
        "payload:nonexistent_tag:value,"
        "payload:onset:missing_field"
    )
    assert _build_csv_converters(header, "1.0") == {}

    # 4. Mock type hint resolution error (NameError / TypeError)
    with patch("typing.get_type_hints", side_effect=NameError("Mock name error")):
        assert _build_csv_converters("payload:onset:value", "1.0") == {}


def test_prepare_annotation_for_save_id_variations():
    # Test lines 254, 257, 261
    ann = create(
        bopp_version="1.0",
        media_id="track:save_id_test",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
        generate_id=False,
    )
    assert getattr(ann, "id", msgspec.UNSET) is msgspec.UNSET

    # 1. generate_id=True, validate_id=True with missing ID (line 257 ensure_annotation_id)
    _prepare_annotation_for_save(ann, validate_id=True, generate_id=True)
    expected_id = compute_annotation_id(ann)
    assert ann.id == expected_id

    # 2. generate_id=True, validate_id=True with existing ID (line 259 validate_annotation_id)
    _prepare_annotation_for_save(ann, validate_id=True, generate_id=True)
    assert ann.id == expected_id

    # 3. generate_id=True, validate_id=False -> sets ID unconditionally (line 254)
    ann_no_id = create(
        bopp_version="1.0",
        media_id="track:save_id_test2",
        payload_kind="onset",
        extent_kind="time",
        time=[0.2],
        value=[2],
        generate_id=False,
    )
    _prepare_annotation_for_save(ann_no_id, validate_id=False, generate_id=True)
    assert ann_no_id.id == compute_annotation_id(ann_no_id)

    # 4. generate_id=False, validate_id=True with valid existing ID (line 261)
    _prepare_annotation_for_save(ann, validate_id=True, generate_id=False)
    assert ann.id == expected_id
