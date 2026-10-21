from unittest.mock import patch

import msgspec
import pandas as pd
import pytest

from bopp.core import compute_annotation_id, create
from bopp.exceptions import BoppArgumentError, BoppRegistryError, BoppValidationError
from bopp.io import (
    AnnotationDataFrame,
    _build_csv_converters,
    _parse_complex_cell,
    _prepare_annotation_for_save,
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.extent.times import Times
from bopp.models.v1.grouping import Grouping
from bopp.models.v1.payload.onset import OnsetPayload
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


def test_csv_roundtrip_with_groupings(tmp_path):
    ann = Annotation(
        media_id="track:csv_groupings",
        bopp_version="1.0",
        extent=Times(time=[0.1, 0.4]),
        payload=OnsetPayload(value=[1, 2]),
        groupings=[
            Grouping(name="voices", ids=[0, 1], labels={"v1": 0, "v2": 1}),
            Grouping(name="phrases", ids=[10, 10]),
        ],
    )
    file_path = tmp_path / "test_groupings.csv"

    save_bopp_csv(ann, file_path, backend="polars")

    loaded = load_bopp_csv(file_path, backend="polars")
    assert loaded.media_id == ann.media_id
    assert len(loaded.groupings) == 2
    assert loaded.groupings[0].name == "voices"
    assert loaded.groupings[0].ids == [0, 1]
    assert loaded.groupings[0].labels == {"v1": 0, "v2": 1}
    assert loaded.groupings[1].name == "phrases"
    assert loaded.groupings[1].ids == [10, 10]


def test_csv_roundtrip_with_groupings_pandas(tmp_path):
    ann = Annotation(
        media_id="track:csv_groupings_pd",
        bopp_version="1.0",
        extent=Times(time=[0.1, 0.4]),
        payload=OnsetPayload(value=[1, 2]),
        groupings=[
            Grouping(name="voices", ids=[0, 1]),
        ],
    )
    file_path = tmp_path / "test_groupings_pd.csv"

    save_bopp_csv(ann, file_path, backend="pandas")

    loaded = load_bopp_csv(file_path, backend="pandas")
    assert loaded.media_id == ann.media_id
    assert len(loaded.groupings) == 1
    assert loaded.groupings[0].name == "voices"
    assert loaded.groupings[0].ids == [0, 1]


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


def test_csv_roundtrip_object_payload_polars(tmp_path):
    pytest.importorskip("polars")

    ann = create(
        bopp_version="1.0",
        media_id="track:csv_obj_polars",
        payload_kind="object",
        value=[{"color": "brown"}, {"color": "white"}],
    )
    file_path = tmp_path / "test_obj_polars.csv"

    # Must succeed with Polars without raising write_csv nested type errors
    save_bopp_csv(ann, file_path, backend="polars")

    loaded = load_bopp_csv(file_path)
    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == "object"
    assert getattr(loaded.payload, "value") == [{"color": "brown"}, {"color": "white"}]


def test_csv_roundtrip_nested_extent_polars(tmp_path):
    pytest.importorskip("polars")

    ann = create(
        bopp_version="1.0",
        media_id="track:csv_extent_polars",
        extent_kind="score_interval",
        payload_kind="chord",
        quarter=[(0, 1), (1, 2)],
        duration=[(1, 2), (1, 4)],
        value=["C:maj", "G:maj"],
    )
    file_path = tmp_path / "test_extent_polars.csv"

    # Must succeed with Polars backend
    save_bopp_csv(ann, file_path, backend="polars")

    loaded = load_bopp_csv(file_path)
    assert loaded.media_id == ann.media_id
    assert getattr(loaded.payload, "value") == ["C:maj", "G:maj"]
    assert getattr(loaded.extent, "quarter") == [(0, 1), (1, 2)]
    assert getattr(loaded.extent, "duration") == [(1, 2), (1, 4)]


@pytest.mark.parametrize(
    "ann_factory",
    [
        # score_quarter extent with fractional tuple (quarter)
        lambda: create(
            bopp_version="1.0",
            media_id="track:score_quarter_polars",
            extent_kind="score_quarter",
            payload_kind="chord",
            quarter=[(0, 1), (3, 4)],
            value=["C:maj", "G:7"],
        ),
        # pitch_contour payload with nested dictionaries
        lambda: create(
            bopp_version="1.0",
            media_id="track:pitch_contour_polars",
            extent_kind="time",
            payload_kind="pitch_contour",
            time=[0.0, 0.01],
            value=[
                {"index": 0, "frequency": 440.0, "voicing": 1.0},
                {"index": 1, "frequency": 442.5, "voicing": 1.0},
            ],
        ),
        # ext payload with scalar ext_schema attribute and list of nested dicts
        lambda: create(
            bopp_version="1.0",
            media_id="track:ext_polars",
            extent_kind="time",
            payload_kind="ext",
            ext_schema="https://example.com/schema.json",
            time=[0.1, 0.5],
            value=[{"feature": "val1", "flags": [1, 2]}, {"feature": "val2", "flags": [3]}],
        ),
    ],
)
def test_csv_roundtrip_complex_types_polars(tmp_path, ann_factory):
    pytest.importorskip("polars")

    ann = ann_factory()
    payload_tag = _get_tag(ann.payload)
    file_path = tmp_path / f"test_complex_{payload_tag}_polars.csv"

    save_bopp_csv(ann, file_path, backend="polars")
    loaded = load_bopp_csv(file_path, backend="polars")

    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == _get_tag(ann.payload)
    if ann.extent is not None:
        assert _get_tag(loaded.extent) == _get_tag(ann.extent)

    for facet in ("payload", "extent"):
        orig_facet = getattr(ann, facet, None)
        loaded_facet = getattr(loaded, facet, None)
        if orig_facet is not None:
            for field in msgspec.structs.fields(orig_facet):
                assert getattr(loaded_facet, field.name) == getattr(orig_facet, field.name)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"payload_kind": "onset", "extent_kind": "time", "time": [0, 1.5], "value": ["a", "b"]},
        {
            "payload_kind": "tag_open",
            "extent_kind": "time_interval",
            "time": [0.0, 1.0],
            "duration": [1, 1.5],
            "value": ["a", "b"],
        },
        {"payload_kind": "tempo", "extent_kind": "time", "time": [0.0, 1.0], "value": [120, 90.5]},
        {
            "payload_kind": "note_hz",
            "extent_kind": "time_interval",
            "time": [0.0, 1.0],
            "duration": [1.0, 1.0],
            "value": [440, 220.5],
        },
        {
            "payload_kind": "note_midi",
            "extent_kind": "midi_interval",
            "tick": [0, 480.5],
            "duration": [240, 240.5],
            "value": [60, 61.5],
        },
        {
            "payload_kind": "tag_open",
            "extent_kind": "time_frequency_box",
            "time": [0.0, 1.0],
            "duration": [1, 0.5],
            "freq_min": [100, 200.5],
            "freq_max": [1000, 2000.5],
            "value": ["a", "b"],
        },
        {
            "payload_kind": "tag_open",
            "extent_kind": "pixel_box",
            "x": [0, 1.5],
            "width": [10, 2.5],
            "y": [0, 1],
            "height": [5, 5.5],
            "value": ["a", "b"],
        },
        {
            "payload_kind": "tag_open",
            "extent_kind": "time",
            "confidence_kind": "likelihood",
            "time": [0.0, 1.0],
            "value": ["a", "b"],
            "confidence": [1, 0.5],
        },
    ],
)
def test_csv_roundtrip_polars_mixed_int_float(tmp_path, kwargs):
    """Number columns holding ints and floats are written and read back as floats."""
    ann = create(media_id="track:mixed", generate_id=False, **kwargs)
    file_path = tmp_path / "mixed.csv"

    save_bopp_csv(ann, file_path, backend="polars", generate_id=False, validate_id=False)
    loaded = load_bopp_csv(file_path, backend="polars", validate_id=False)

    for facet in ("payload", "extent", "confidence"):
        original = getattr(ann, facet, None)
        if not isinstance(original, msgspec.Struct):
            continue
        restored = getattr(loaded, facet)
        for field in msgspec.structs.fields(original):
            values = getattr(original, field.name)
            if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                assert getattr(restored, field.name) == [float(v) for v in values], f"{facet}.{field.name}"
                assert all(type(v) is float for v in getattr(restored, field.name)), f"{facet}.{field.name}"


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
    header = {"media_id": "track:col_rules", "bopp_version": "1.0"}

    # Rule 1: Column format does not have 3 colon-separated parts or group.N.ids
    df_invalid_format = pd.DataFrame({"payload:onset": [1, 2]})
    with pytest.raises(BoppValidationError, match="Invalid column format"):
        from_dataframe(df_invalid_format, header)

    # Rule 2: Facet has multiple types
    df_mixed_types = pd.DataFrame({
        "extent:time:time": [0.1, 0.2],
        "extent:time_interval:start": [0.1, 0.2],
        "payload:onset:value": [1, 1],
    })
    with pytest.raises(BoppValidationError, match="Mixed types for facet 'extent'"):
        from_dataframe(df_mixed_types, header)

    # Rule 3: Unrecognized facet or type
    df_invalid_type = pd.DataFrame({
        "payload:nonexistent_type:value": [1, 2]
    })
    with pytest.raises(BoppValidationError, match="Unrecognized type 'nonexistent_type'"):
        from_dataframe(df_invalid_type, header)

    # Rule 4: Field does not exist on target type struct
    df_invalid_field = pd.DataFrame({
        "payload:onset:invalid_field_name": [1, 2]
    })
    with pytest.raises(BoppValidationError, match="Field 'invalid_field_name' in column .* does not exist"):
        from_dataframe(df_invalid_field, header)


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

    mock_ann_df = AnnotationDataFrame(df=MockUnsupportedDataFrame(), header={"media_id": "test"})
    with patch("bopp.io.to_dataframe", return_value=mock_ann_df):
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

    mock_ann_df = AnnotationDataFrame(
        df=polars.DataFrame({"payload:onset:value": [1]}),
        header={"media_id": "track:polars_test"},
    )
    with patch("bopp.io.to_dataframe", return_value=mock_ann_df):
        save_bopp_csv(ann, file_path)

    assert file_path.exists()


def test_load_bopp_csv_without_frontmatter(tmp_path):
    # Test handling of legacy or unfrontmattered CSV files
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


def test_parse_complex_and_str_cell():
    # Empty or whitespace strings evaluate safely to None for complex fields
    assert _parse_complex_cell(None) is None  # type: ignore[arg-type]
    assert _parse_complex_cell("   ") is None

    # Valid container literal strings are evaluated via JSON or ast.literal_eval
    assert _parse_complex_cell("[1, 2, 3]") == [1, 2, 3]
    assert _parse_complex_cell('{"key": "val"}') == {"key": "val"}

    # Plain string literals not formatted as containers are left untouched
    assert _parse_complex_cell("unbracketed_val") == "unbracketed_val"


def test_build_csv_converters_error_and_edge_cases():
    # Empty header line returns empty converters map
    assert _build_csv_converters("", "1.0") == {}

    # Unknown schema version safely yields no converters
    assert _build_csv_converters("payload:onset:value", "nonexistent_version") == {}

    # Columns that are malformed, outside recognized facets, or missing fields are ignored
    header = (
        "not_three_parts,"
        "unknown_facet:foo:bar,"
        "payload:nonexistent_tag:value,"
        "payload:onset:missing_field"
    )
    assert _build_csv_converters(header, "1.0") == {}

    # Valid complex and string columns map to respective parsers
    converters = _build_csv_converters("extent:score_interval:quarter,payload:tag_open:value", "1.0")
    assert converters["extent:score_interval:quarter"] is _parse_complex_cell


def test_prepare_annotation_for_save_id_variations():
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

    # Generate missing ID when generate_id=True and validate_id=True
    _prepare_annotation_for_save(ann, validate_id=True, generate_id=True)
    expected_id = compute_annotation_id(ann)
    assert ann.id == expected_id

    # Validate existing ID without regenerating when generate_id=True and validate_id=True
    _prepare_annotation_for_save(ann, validate_id=True, generate_id=True)
    assert ann.id == expected_id

    # Unconditionally set ID when generate_id=True and validate_id=False
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

    # Strictly validate existing ID when generate_id=False and validate_id=True
    _prepare_annotation_for_save(ann, validate_id=True, generate_id=False)
    assert ann.id == expected_id
