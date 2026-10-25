import builtins
import io
from unittest.mock import patch

import msgspec
import pandas as pd
import pytest

from bopp.core import compute_annotation_id, create
from bopp.exceptions import (
    BoppArgumentError,
    BoppIOError,
    BoppValidationError,
)
from bopp.io import (
    _build_csv_converters,
    _open_or_yield_text_stream,
    _parse_complex_cell,
    _prepare_annotation_for_save,
    _read_bytes_from_source,
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
from bopp.models.v1.payload.beat import BeatPositionPayload
from bopp.util import _get_tag, from_dataframe


def test_json_roundtrip(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:json_test",
        payload_kind="beat",
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


def test_json_stream_and_string_io():
    ann = create(
        bopp_version="1.0",
        media_id="track:json_stream_test",
        payload_kind="beat",
        extent_kind="time",
        time=[0.1, 0.5],
        value=[1, 1],
    )

    # 1. Text stream (StringIO)
    buf_text = io.StringIO()
    save_bopp_json(ann, buf_text)
    buf_text.seek(0)
    assert not buf_text.closed
    loaded_from_text = load_bopp_json(buf_text)
    assert loaded_from_text.id == ann.id
    assert not buf_text.closed

    # 2. Raw JSON string via StringIO
    raw_json_str = buf_text.getvalue()
    loaded_from_str = load_bopp_json(io.StringIO(raw_json_str))
    assert loaded_from_str.id == ann.id

    # 3. Binary stream (BytesIO)
    buf_bin = io.BytesIO()
    save_bopp_json(ann, buf_bin)
    buf_bin.seek(0)
    assert not buf_bin.closed
    loaded_from_bin = load_bopp_json(buf_bin)
    assert loaded_from_bin.id == ann.id
    assert not buf_bin.closed

    # 4. Raw bytes
    loaded_from_bytes = load_bopp_json(buf_bin.getvalue())
    assert loaded_from_bytes.id == ann.id


def test_msgpack_stream_and_bytes_io():
    ann = create(
        bopp_version="1.0",
        media_id="track:msgpack_stream_test",
        payload_kind="beat",
        extent_kind="time",
        time=[0.2, 0.6],
        value=[1, 1],
    )

    # Binary stream (BytesIO)
    buf = io.BytesIO()
    save_bopp_msgpack(ann, buf)
    assert not buf.closed
    buf.seek(0)

    loaded_from_stream = load_bopp_msgpack(buf)
    assert loaded_from_stream.id == ann.id
    assert not buf.closed

    # Raw bytes
    loaded_from_bytes = load_bopp_msgpack(buf.getvalue())
    assert loaded_from_bytes.id == ann.id


def test_csv_stream_and_string_io():
    ann = create(
        bopp_version="1.0",
        media_id="track:csv_stream_test",
        payload_kind="beat",
        extent_kind="time",
        time=[0.1, 0.4],
        value=[1, 1],
        sandbox={"user_note": "csv_stream"},
    )

    # 1. Text stream (StringIO)
    buf = io.StringIO()
    save_bopp_csv(ann, buf)
    assert not buf.closed
    buf.seek(0)

    loaded_stream = load_bopp_csv(buf)
    assert loaded_stream.id == ann.id
    assert not buf.closed

    # 2. Raw CSV string via StringIO
    raw_csv = buf.getvalue()
    loaded_str = load_bopp_csv(io.StringIO(raw_csv))
    assert loaded_str.id == ann.id
    assert getattr(loaded_str, "sandbox", None) == {"user_note": "csv_stream"}

    # 3. Raw CSV string via StringIO with Pandas backend
    loaded_pandas = load_bopp_csv(io.StringIO(raw_csv), backend="pandas")
    assert loaded_pandas.id == ann.id


def test_load_bopp_json_missing_id(tmp_path):
    # Valid JSON without an 'id' field
    json_data = '{"bopp_version": "1.0", "media_id": "track:123", "payload": {"payload_type": "beat", "time": [0.1], "value": [1]}}'
    file_path = tmp_path / "missing_id.json"
    file_path.write_text(json_data, encoding="utf-8")

    loaded = load_bopp_json(file_path, validate_id=True)
    assert getattr(loaded, "id", msgspec.UNSET) is msgspec.UNSET


def test_load_bopp_json_mismatched_id_error(tmp_path):
    # Valid UUID string format to pass msgspec schema validation, but computed ID mismatch
    invalid_uuid = "00000000-0000-5000-8000-000000000000"
    json_data = f'{{"bopp_version": "1.0", "media_id": "track:123", "id": "{invalid_uuid}", "payload": {{"payload_type": "beat", "time": [0.1], "value": [1]}}}}'
    file_path = tmp_path / "mismatched_id.json"
    file_path.write_text(json_data, encoding="utf-8")

    with pytest.raises(BoppValidationError, match="Annotation ID mismatch"):
        load_bopp_json(file_path, validate_id=True)


def test_load_bopp_json_invalid_schema(tmp_path):
    # Valid JSON but invalid structure (missing required 'media_id')
    invalid_json_data = '{"bopp_version": "1.0", "payload": {"payload_type": "beat", "time": [0.1], "value": [1]}}'
    file_path = tmp_path / "invalid.json"
    file_path.write_text(invalid_json_data, encoding="utf-8")

    with pytest.raises((msgspec.ValidationError, BoppValidationError)):
        load_bopp_json(file_path)


def test_msgpack_roundtrip(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:msgpack_test",
        payload_kind="beat",
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
        payload_kind="beat",
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
        payload=BeatPositionPayload(value=[1, 2]),
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
        payload=BeatPositionPayload(value=[1, 2]),
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
    assert loaded.payload.value == [{"color": "brown"}, {"color": "white"}]


def test_csv_roundtrip_nested_extent_polars(tmp_path):
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
    assert loaded.payload.value == ["C:maj", "G:maj"]
    assert loaded.extent.quarter == [(0, 1), (1, 2)]
    assert loaded.extent.duration == [(1, 2), (1, 4)]


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
        {"payload_kind": "marker", "extent_kind": "time", "time": [0, 1.5], "description": "marker_events"},
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
        "payload:beat:nonsense,extent:time:time\n"
        "1,0.1\n"
    )
    file_path = tmp_path / "invalid.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    with pytest.raises(BoppValidationError, match="Field"):
        load_bopp_csv(file_path)


def test_csv_column_validation_rules():
    header = {"media_id": "track:col_rules", "bopp_version": "1.0"}

    # Rule 1: Column format does not have 3 colon-separated parts or group.N.ids
    df_invalid_format = pd.DataFrame({"payload:beat": [1, 2]})
    with pytest.raises(BoppValidationError, match="Invalid column format"):
        from_dataframe(df_invalid_format, header)

    # Rule 2: Facet has multiple types
    df_mixed_types = pd.DataFrame({
        "extent:time:time": [0.1, 0.2],
        "extent:time_interval:start": [0.1, 0.2],
        "payload:beat:value": [1, 1],
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
        "payload:beat:invalid_field_name": [1, 2]
    })
    with pytest.raises(BoppValidationError, match="Field 'invalid_field_name' in column .* does not exist"):
        from_dataframe(df_invalid_field, header)


def test_save_and_load_with_validate_id_false(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:no_id_val",
        payload_kind="beat",
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


def test_save_bopp_csv_pandas(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:pandas_test",
        payload_kind="beat",
        extent_kind="time",
        time=[0.1],
        value=[1],
    )
    file_path = tmp_path / "pandas_test.csv"
    save_bopp_csv(ann, file_path, backend="pandas")
    loaded = load_bopp_csv(file_path, backend="pandas")
    assert loaded.id == ann.id


def test_load_bopp_csv_without_frontmatter(tmp_path):
    # Test handling of legacy or unfrontmattered CSV files
    csv_content = "payload:beat:value,extent:time:time\n1,0.1\n"
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
        "payload:beat:value,extent:time:time\n"
        "1,0.1\n"
    )
    file_path = tmp_path / "extra_col.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    loaded = load_bopp_csv(file_path, validate_id=True)
    assert loaded.media_id == "track:extra_col"


def test_load_bopp_csv_crlf_newlines(tmp_path):
    """Verify CSV files with CRLF (\\r\\n) line endings load correctly across platforms."""
    csv_content = (
        b"# ---\r\n"
        b'# bopp_version = "1.0"\r\n'
        b'# media_id = "track:crlf_test"\r\n'
        b"# ---\r\n"
        b"payload:beat:value,extent:time:time\r\n"
        b"1,0.1\r\n"
        b"2,0.5\r\n"
    )
    file_path = tmp_path / "crlf.csv"
    file_path.write_bytes(csv_content)

    loaded = load_bopp_csv(file_path, validate_id=False)
    assert loaded.media_id == "track:crlf_test"
    assert loaded.payload.value == [1, 2]
    assert loaded.extent.time == [0.1, 0.5]


def test_parse_complex_and_str_cell():
    # Empty or whitespace strings evaluate safely to None for complex fields
    assert _parse_complex_cell(None) is None  # type: ignore[arg-type]
    assert _parse_complex_cell("   ") is None

    # Valid container literal strings are evaluated via JSON or ast.literal_eval
    assert _parse_complex_cell("[1, 2, 3]") == [1, 2, 3]
    assert _parse_complex_cell('{"key": "val"}') == {"key": "val"}

    # Plain string literals not formatted as containers are left untouched
    assert _parse_complex_cell("unbracketed_val") == "unbracketed_val"


def test_parse_complex_cell_non_str():
    # Non-string, non-None values return None safely
    assert _parse_complex_cell(123) is None  # type: ignore[arg-type]
    assert _parse_complex_cell(object()) is None  # type: ignore[arg-type]


def test_build_csv_converters_error_and_edge_cases():
    # Empty header line returns empty converters map
    assert _build_csv_converters("", "1.0") == {}

    # Unknown schema version safely yields no converters
    assert _build_csv_converters("payload:beat:value", "nonexistent_version") == {}

    # Columns that are malformed, outside recognized facets, or missing fields are ignored
    header = (
        "not_three_parts,"
        "unknown_facet:foo:bar,"
        "payload:nonexistent_tag:value,"
        "payload:beat:missing_field"
    )
    assert _build_csv_converters(header, "1.0") == {}

    # Valid complex and string columns map to respective parsers
    converters = _build_csv_converters("extent:score_interval:quarter,payload:tag_open:value", "1.0")
    assert converters["extent:score_interval:quarter"] is _parse_complex_cell


def test_prepare_annotation_for_save_id_variations():
    ann = create(
        bopp_version="1.0",
        media_id="track:save_id_test",
        payload_kind="beat",
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

    # Unconditionally set/overwrite ID when generate_id=True and validate_id=False
    ann_no_id = create(
        bopp_version="1.0",
        media_id="track:save_id_test2",
        payload_kind="beat",
        extent_kind="time",
        time=[0.2],
        value=[2],
        generate_id=False,
    )
    _prepare_annotation_for_save(ann_no_id, validate_id=False, generate_id=True)
    assert ann_no_id.id == compute_annotation_id(ann_no_id)

    ann_no_id.payload.value = [3]
    _prepare_annotation_for_save(ann_no_id, validate_id=False, generate_id=True)
    assert ann_no_id.id == compute_annotation_id(ann_no_id)

    # Strictly validate existing ID when generate_id=False and validate_id=True
    _prepare_annotation_for_save(ann, validate_id=True, generate_id=False)
    assert ann.id == expected_id


def test_read_bytes_from_source_invalid():
    with pytest.raises(BoppIOError, match="Unsupported source type for binary read"):
        _read_bytes_from_source(12345)  # type: ignore[arg-type]


def test_read_bytes_from_source_buffer_types():
    data = b'{"bopp_version": "1.0", "media_id": "track:buf_test", "payload": {"payload_type": "beat", "time": [0.1], "value": [1]}}'

    # bytearray
    loaded_bytearray = load_bopp_json(bytearray(data), validate_id=False)
    assert loaded_bytearray.media_id == "track:buf_test"

    # memoryview
    mv = memoryview(data)
    assert _read_bytes_from_source(mv) == data
    loaded_mv = load_bopp_json(mv, validate_id=False)
    assert loaded_mv.media_id == "track:buf_test"

    # text stream returning str from read()
    text_stream = io.StringIO(data.decode("utf-8"))
    assert _read_bytes_from_source(text_stream) == data

    # binary stream returning bytes from read()
    bin_stream = io.BytesIO(data)
    assert _read_bytes_from_source(bin_stream) == data


def test_load_bopp_csv_invalid_source_type():
    with pytest.raises(BoppIOError, match="Unsupported source type for text stream"):
        load_bopp_csv(12345)  # type: ignore[arg-type]

    with (
        pytest.raises(BoppIOError, match="Unsupported source type for text stream"),
        _open_or_yield_text_stream(b"binary data"),  # type: ignore[arg-type]
    ):
        pass


class _NonSeekableTextStream:
    """A text stream wrapper that deliberately lacks a 'seek' method."""

    def __init__(self, content: str) -> None:
        self._lines = content.splitlines(keepends=True)
        self._pos = 0

    def readline(self) -> str:
        if self._pos < len(self._lines):
            line = self._lines[self._pos]
            self._pos += 1
            return line
        return ""

    def read(self) -> str:
        content = "".join(self._lines[self._pos:])
        self._pos = len(self._lines)
        return content


def test_load_bopp_csv_non_seekable_stream():
    csv_content = (
        "# ---\n"
        '# bopp_version = "1.0"\n'
        '# media_id = "track:non_seekable"\n'
        "# ---\n"
        "payload:beat:value,extent:time:time\n"
        "1,0.1\n"
        "2,0.5\n"
    )
    stream = _NonSeekableTextStream(csv_content)
    assert not hasattr(stream, "seek")

    loaded = load_bopp_csv(stream, validate_id=False)  # type: ignore[arg-type]
    assert loaded.media_id == "track:non_seekable"
    assert loaded.payload.value == [1, 2]
    assert loaded.extent.time == [0.1, 0.5]


def test_load_bopp_csv_unsupported_backend():
    csv_content = (
        "# ---\n"
        '# bopp_version = "1.0"\n'
        '# media_id = "track:invalid_backend"\n'
        "# ---\n"
        "payload:beat:value,extent:time:time\n"
        "1,0.1\n"
    )
    with pytest.raises(BoppArgumentError, match="Unsupported backend: unknown"):
        load_bopp_csv(io.StringIO(csv_content), backend="unknown")  # type: ignore[arg-type]


def test_load_bopp_csv_pandas_missing():
    csv_content = (
        "# ---\n"
        '# bopp_version = "1.0"\n'
        '# media_id = "track:pandas_missing"\n'
        "# ---\n"
        "payload:beat:value,extent:time:time\n"
        "1,0.1\n"
    )
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        pytest.raises(BoppIOError, match="pandas is required for backend='pandas'"),
    ):
        load_bopp_csv(io.StringIO(csv_content), backend="pandas")


def test_csv_roundtrip_marker_payload(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:marker_csv",
        payload_kind="marker",
        extent_kind="time",
        time=[0.1, 0.5, 1.2],
        description="cues",
        sandbox={"note": "marker_roundtrip"},
    )
    for backend in ("polars", "pandas"):
        file_path = tmp_path / f"marker_{backend}.csv"
        save_bopp_csv(ann, file_path, backend=backend)
        loaded = load_bopp_csv(file_path, backend=backend)
        assert loaded.media_id == ann.media_id
        assert loaded.payload.description == "cues"
        assert loaded.extent.time == [0.1, 0.5, 1.2]
        assert loaded.id == ann.id


def test_csv_roundtrip_marker_payload_without_extent(tmp_path):
    ann = create(
        bopp_version="1.0",
        media_id="track:marker_no_extent",
        payload_kind="marker",
        description="global_label",
    )
    for backend in ("polars", "pandas"):
        file_path = tmp_path / f"marker_no_extent_{backend}.csv"
        save_bopp_csv(ann, file_path, backend=backend)
        loaded = load_bopp_csv(file_path, backend=backend)
        assert loaded.media_id == ann.media_id
        assert loaded.payload.description == "global_label"
        assert loaded.extent is msgspec.UNSET
        assert loaded.id == ann.id
