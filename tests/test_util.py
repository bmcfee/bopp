import enum
from typing import Annotated, Any, Literal
from unittest.mock import patch

import msgspec
import pandas as pd
import polars as pl
import pytest

from bopp.core import create
from bopp.exceptions import (
    BoppArgumentError,
    BoppArrayError,
    BoppIOError,
    BoppValidationError,
)
from bopp.io import _type_to_polars_dtype
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.confidence.likelihood import LikelihoodConfidence
from bopp.models.v1.extent.score_interval import ScoreInterval
from bopp.models.v1.extent.time import Time
from bopp.models.v1.grouping import Grouping
from bopp.models.v1.payload.beat import BeatPositionPayload
from bopp.models.v1.payload.marker import MarkerPayload
from bopp.models.v1.payload.object import ObjectPayload
from bopp.registries import FACET_REGISTRY_KEYS, get_registry
from bopp.util import (
    AnnotationDataFrame,
    _extract_facet_data,
    _get_tag,
    _polars_column_dtype,
    _validate_dataframe_columns,
    from_dataframe,
    to_dataframe,
)


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


def test_extract_facet_data_scalar_attributes():
    class DummyFacet(msgspec.Struct, tag="beat"):
        value: list[int]
        sample_rate: float
        payload_type: str = "beat"
        optional_attr: str | None = None
        unset_attr: str = msgspec.UNSET

    facet = DummyFacet(value=[1, 2], sample_rate=44100.0)
    target_dict: dict[str, list[int]] = {}
    attrs_dict: dict[str, dict[str, float]] = {}

    _extract_facet_data(facet, "payload", target_dict, attrs_dict)

    assert target_dict == {"payload:beat:value": [1, 2]}
    assert attrs_dict == {"payload": {"sample_rate": 44100.0}}

    # When struct is None or UNSET, function returns early
    _extract_facet_data(None, "payload", target_dict, attrs_dict)
    _extract_facet_data(msgspec.UNSET, "payload", target_dict, attrs_dict)


def test_extract_facet_data_encode_complex():
    payload = ObjectPayload(value=[{"a": 1}, {"b": 2}])
    target_dict: dict[str, list[Any]] = {}
    attrs_dict: dict[str, Any] = {}

    _extract_facet_data(
        payload,
        "payload",
        target_dict,
        attrs_dict,
        encode_complex=True,
        complex_fields={"value"},
    )
    assert target_dict["payload:object:value"] == ['{"a":1}', '{"b":2}']


def test_to_dataframe_encode_complex():
    ann = Annotation(
        media_id="track:complex_test",
        bopp_version="1.0",
        payload=ObjectPayload(value=[{"k": "v"}]),
        extent=ScoreInterval(quarter=[[0, 1]], duration=[[1, 2]]),
    )

    # With encode_complex=False, columns contain raw Python lists/dicts
    ann_df_raw = to_dataframe(ann, backend="pandas", encode_complex=False)
    assert isinstance(ann_df_raw, AnnotationDataFrame)
    assert ann_df_raw.df["payload:object:value"][0] == {"k": "v"}
    assert ann_df_raw.df["extent:score_interval:quarter"][0] == [0, 1]

    # With encode_complex=True, columns contain JSON-encoded strings
    ann_df_encoded = to_dataframe(ann, backend="pandas", encode_complex=True)
    assert ann_df_encoded.df["payload:object:value"][0] == '{"k":"v"}'
    assert ann_df_encoded.df["extent:score_interval:quarter"][0] == "[0,1]"
    assert ann_df_encoded.df["extent:score_interval:duration"][0] == "[1,2]"


def test_to_dataframe_polars_empty_schema_fallback():
    ann = Annotation(
        media_id="track:untyped",
        bopp_version="1.0",
        payload=ObjectPayload(value=[{"a": 1}]),
    )
    ann_df = to_dataframe(ann, backend="polars", encode_complex=False)
    assert isinstance(ann_df.df, pl.DataFrame)
    assert ann_df.df["payload:object:value"][0] == {"a": 1}


def test_validate_dataframe_columns_invalid_format():
    """Verify that DataFrame column header validation rejects column names that do not
    adhere to the required naming format.
    """
    invalid_columns = [
        "invalid_column_no_colons",
        "extent:time",  # Too few parts (2)
        "payload:beat:value:extra",  # Too many parts (4)
        ":::",  # Empty parts
        "group.notanint.ids",  # Invalid grouping index
        "group.0.invalid_field",  # Field must be 'ids'
    ]
    for col in invalid_columns:
        with pytest.raises(BoppValidationError, match="Invalid"):
            _validate_dataframe_columns([col], "1.0")


def test_validate_dataframe_columns_rules():
    """Verify DataFrame column validation rules 2, 3, and 4."""
    # Rule 2: Mixed types for facet
    with pytest.raises(BoppValidationError, match="Mixed types for facet 'payload'"):
        _validate_dataframe_columns(["payload:tag:value", "payload:beat:value"], "1.0")

    # Rule 3a: Invalid facet
    with pytest.raises(BoppValidationError, match="Invalid facet 'unknown_facet'"):
        _validate_dataframe_columns(["unknown_facet:beat:value"], "1.0")

    # Rule 3b: Unrecognized type tag in registry
    with pytest.raises(BoppValidationError, match="Unrecognized type 'unknown_tag'"):
        _validate_dataframe_columns(["payload:unknown_tag:value"], "1.0")

    # Rule 4: Field does not exist on target type struct
    with pytest.raises(
        BoppValidationError,
        match="Field 'nonexistent_field' in column .* does not exist",
    ):
        _validate_dataframe_columns(["payload:beat:nonexistent_field"], "1.0")


def test_to_dataframe_partial_facets():
    # Test annotation without extent or confidence
    ann = create(
        bopp_version="1.0",
        media_id="track:no_extent",
        payload_type="beat",
        value=[1, 1],
    )
    ann_df = to_dataframe(ann, backend="pandas")
    assert isinstance(ann_df, AnnotationDataFrame)
    assert isinstance(ann_df.df, pd.DataFrame)
    assert ann_df.header["media_id"] == "track:no_extent"
    assert not any(col.startswith("extent:") for col in ann_df.df.columns)
    assert not any(col.startswith("confidence:") for col in ann_df.df.columns)

    # Reconstitute from AnnotationDataFrame with missing extent & confidence columns
    reconstructed = from_dataframe(ann_df)
    assert reconstructed.media_id == "track:no_extent"
    assert getattr(reconstructed, "extent", None) is msgspec.UNSET


def test_to_dataframe_filters_tag_discriminators_and_unset_metadata():
    """Verify that `to_dataframe` correctly extracts array data and metadata headers while
    omitting tag discriminator fields (e.g. `payload_type`) and UNSET optional fields.
    """
    ann = Annotation(
        media_id="track:skip_test",
        bopp_version="1.0",
        metadata=msgspec.UNSET,
        extent=Time(time=[0.1, 0.2]),
        payload=BeatPositionPayload(value=[1, 1]),
        confidence=LikelihoodConfidence(confidence=[0.8, 0.9]),
    )

    df, header = to_dataframe(ann, backend="pandas")

    # Verify parallel array data columns are created correctly
    assert list(df["extent:time:time"]) == [0.1, 0.2]
    assert list(df["payload:beat:value"]) == [1, 1]
    assert list(df["confidence:likelihood:confidence"]) == [0.8, 0.9]

    # Verify discriminator tag fields are excluded from DataFrame columns
    assert "extent:time:extent_type" not in df.columns
    assert "payload:beat:payload_type" not in df.columns
    assert "confidence:likelihood:confidence_type" not in df.columns

    # Verify UNSET metadata is not in header
    assert "metadata" not in header


def test_to_dataframe_covers_extent_and_confidence_branches():
    """Verify that to_dataframe extracts parallel arrays and attributes when both extent and confidence exist."""
    ann = create(
        bopp_version="1.0",
        media_id="track:facets_coverage",
        payload_type="beat",
        extent_type="time",
        confidence_type="likelihood",
        time=[0.1, 0.2],
        value=[1, 1],
        confidence=[0.85, 0.95],
    )
    df, _header = to_dataframe(ann, backend="pandas")
    assert "extent:time:time" in df.columns
    assert "confidence:likelihood:confidence" in df.columns


def test_groupings_to_and_from_dataframe_pandas():
    ann = Annotation(
        media_id="track:grouping_pandas",
        bopp_version="1.0",
        payload=BeatPositionPayload(value=[1, 2, 3]),
        extent=Time(time=[0.1, 0.2, 0.3]),
        groupings=[
            Grouping(name="voices", ids=[0, 1, 0], labels={"soprano": 0, "alto": 1}),
            Grouping(name="measures", ids=[1, 1, 2]),
        ],
    )

    ann_df = to_dataframe(ann, backend="pandas")
    df, header = ann_df.df, ann_df.header

    assert "group.0.ids" in df.columns
    assert "group.1.ids" in df.columns
    assert list(df["group.0.ids"]) == [0, 1, 0]
    assert list(df["group.1.ids"]) == [1, 1, 2]

    assert len(header["groupings"]) == 2
    assert header["groupings"][0]["name"] == "voices"
    assert header["groupings"][0]["labels"] == {"soprano": 0, "alto": 1}
    assert header["groupings"][1]["name"] == "measures"

    # Reconstruct
    reconstructed = from_dataframe(ann_df)
    assert reconstructed.media_id == ann.media_id
    assert len(reconstructed.groupings) == 2
    assert reconstructed.groupings[0].name == "voices"
    assert reconstructed.groupings[0].ids == [0, 1, 0]
    assert reconstructed.groupings[0].labels == {"soprano": 0, "alto": 1}
    assert reconstructed.groupings[1].name == "measures"
    assert reconstructed.groupings[1].ids == [1, 1, 2]


def test_groupings_to_and_from_dataframe_polars():
    ann = Annotation(
        media_id="track:grouping_polars",
        bopp_version="1.0",
        payload=BeatPositionPayload(value=[1, 2]),
        groupings=[
            Grouping(name="section", ids=[0, 1]),
        ],
    )

    ann_df = to_dataframe(ann, backend="polars")
    assert isinstance(ann_df.df, pl.DataFrame)
    assert "group.0.ids" in ann_df.df.columns
    assert ann_df.df["group.0.ids"].dtype == pl.Int64

    reconstructed = from_dataframe(ann_df)
    assert reconstructed.groupings[0].name == "section"
    assert reconstructed.groupings[0].ids == [0, 1]


def test_groupings_mismatched_length_raises():
    with pytest.raises(BoppArrayError, match="Length mismatch"):
        Annotation(
            media_id="track:mismatch",
            bopp_version="1.0",
            payload=BeatPositionPayload(value=[1, 2]),
            groupings=[
                Grouping(name="voices", ids=[0]),  # length 1 vs 2
            ],
        )


def test_pandas_dataframe_roundtrip():
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_type="beat",
        extent_type="time",
        confidence_type="likelihood",
        time=[0.1, 0.2],
        value=[1, 1],
        confidence=[0.8, 0.9],
        sandbox={"info": "pandas_test"},
    )

    # Test namedtuple return and named fields
    ann_df = to_dataframe(ann, backend="pandas")
    assert isinstance(ann_df, AnnotationDataFrame)
    assert isinstance(ann_df.df, pd.DataFrame)
    assert ann_df.header["media_id"] == "track:1"
    assert ann_df.header["sandbox"] == {"info": "pandas_test"}

    # Test roundtrip using namedtuple directly
    reconstructed = from_dataframe(ann_df)
    assert reconstructed.media_id == ann.media_id
    assert _get_tag(reconstructed.payload) == _get_tag(ann.payload)
    assert getattr(reconstructed, "sandbox", None) == {"info": "pandas_test"}

    # Test roundtrip using direct tuple unpacking (df, header)
    df, header = to_dataframe(ann, backend="pandas")
    reconstructed2 = from_dataframe(df, header)
    assert reconstructed2.media_id == ann.media_id
    assert _get_tag(reconstructed2.payload) == _get_tag(ann.payload)


def test_pandas_pyarrow_dataframe_roundtrip():
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_type="beat",
        extent_type="time",
        confidence_type="likelihood",
        time=[0.1, 0.2],
        value=[1, 1],
        confidence=[0.8, 0.9],
        sandbox={"info": "pyarrow_test"},
    )

    ann_df = to_dataframe(ann, backend="pandas-pyarrow")
    assert isinstance(ann_df.df, pd.DataFrame)
    assert ann_df.header["media_id"] == "track:1"
    assert ann_df.header["sandbox"] == {"info": "pyarrow_test"}

    reconstructed = from_dataframe(ann_df)
    assert reconstructed.media_id == ann.media_id
    assert _get_tag(reconstructed.payload) == _get_tag(ann.payload)
    assert getattr(reconstructed, "sandbox", None) == {"info": "pyarrow_test"}


def test_polars_dataframe_roundtrip():
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_type="beat",
        extent_type="time",
        confidence_type="likelihood",
        time=[0.1, 0.2],
        value=[1, 1],
        confidence=[0.8, 0.9],
        sandbox={"info": "polars_test"},
    )

    df, header = to_dataframe(ann, backend="polars")
    assert isinstance(df, pl.DataFrame)
    assert header["media_id"] == "track:1"
    assert header["sandbox"] == {"info": "polars_test"}

    reconstructed = from_dataframe(df, header)
    assert reconstructed.media_id == ann.media_id
    assert _get_tag(reconstructed.payload) == _get_tag(ann.payload)
    assert getattr(reconstructed, "sandbox", None) == {"info": "polars_test"}


def test_from_dataframe_with_scalar_facet_attributes():
    """Verify from_dataframe accurately restores scalar facet attributes from header across extent, payload, confidence."""

    class ExtentWithScalar(msgspec.Struct, tag_field="extent_type", tag="time"):
        time: list[float]
        origin: str = "start"

    class PayloadWithScalar(msgspec.Struct, tag_field="payload_type", tag="beat"):
        value: list[int]
        source: str = "mic"

    class ConfWithScalar(msgspec.Struct, tag_field="confidence_type", tag="likelihood"):
        confidence: list[float]
        method: str = "softmax"

    class CustomAnnotation(msgspec.Struct):
        media_id: str
        bopp_version: str
        extent: ExtentWithScalar | None = None
        payload: PayloadWithScalar | None = None
        confidence: ConfWithScalar | None = None

    mock_registry = {
        "Annotation": CustomAnnotation,
        "EXTENT_TYPE_REGISTRY": {"time": ExtentWithScalar},
        "PAYLOAD_TYPE_REGISTRY": {"beat": PayloadWithScalar},
        "CONFIDENCE_TYPE_REGISTRY": {"likelihood": ConfWithScalar},
        "COMPLEX_FIELDS_REGISTRY": {},
    }

    with patch("bopp.util.get_registry", return_value=mock_registry):
        df = pd.DataFrame(
            {
                "extent:time:time": [0.1, 0.2],
                "payload:beat:value": [1, 2],
                "confidence:likelihood:confidence": [0.9, 0.95],
            }
        )
        header = {
            "bopp_version": "1.0",
            "media_id": "track:with_scalars",
            "extent": {"origin": "custom_origin"},
            "payload": {"source": "custom_source"},
            "confidence": {"method": "custom_method"},
        }

        reconstructed = from_dataframe(df, header)
        assert reconstructed.extent.origin == "custom_origin"
        assert reconstructed.payload.source == "custom_source"
        assert reconstructed.confidence.method == "custom_method"


def test_from_dataframe_missing_required_payload_facet():
    """Verify that reconstructing an Annotation from a DataFrame fails validation
    if the DataFrame lacks mandatory 'payload:*' columns required by the schema.
    """
    df = pd.DataFrame({"extent:time:time": [0.1, 0.2]})
    header = {
        "bopp_version": "1.0",
        "media_id": "track:no_payload",
    }

    with pytest.raises(
        msgspec.ValidationError, match="Object missing required field `payload`"
    ):
        from_dataframe(df, header)


def test_from_dataframe_missing_media_id():
    df = pd.DataFrame({"payload:beat:value": [1, 2]})
    header = {
        "bopp_version": "1.0",
    }

    with pytest.raises(
        BoppValidationError, match="Missing or invalid required 'media_id'"
    ):
        from_dataframe(df, header)


def test_from_dataframe_extra_attrs():
    # Test metadata and annotated_domain attributes handling in from_dataframe
    df = pd.DataFrame({"extent:time:time": [0.1], "payload:beat:value": [1]})
    header = {
        "bopp_version": "1.0",
        "media_id": "track:extra_attrs",
        "metadata": {"metadata_type": "other", "notes": "nonsense"},
    }

    reconstructed = from_dataframe(df, header)
    assert reconstructed.media_id == "track:extra_attrs"


def test_to_dataframe_raises_ioerror_when_backend_dependency_missing():
    """Verify that `to_dataframe` raises an actionable BoppIOError when attempting
    to use a DataFrame backend whose underlying library is not installed.
    """
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_type="beat",
        extent_type="time",
        time=[0.1],
        value=[1],
    )

    with (
        patch.dict("sys.modules", {"pandas": None}),
        pytest.raises(BoppIOError, match="pandas is required"),
    ):
        to_dataframe(ann, backend="pandas")

    with (
        patch.dict("sys.modules", {"pyarrow": None}),
        pytest.raises(BoppIOError, match="pyarrow is required"),
    ):
        to_dataframe(ann, backend="pandas-pyarrow")


def test_from_dataframe_raises_argument_error_when_dataframe_libraries_missing():
    """Verify that `from_dataframe` raises a BoppArgumentError when required DataFrame
    library (pandas) cannot be imported to inspect the input object.
    """
    df_pd = pd.DataFrame({"payload:beat:value": [1]})
    with (
        patch.dict("sys.modules", {"pandas": None}),
        pytest.raises(BoppArgumentError, match="Unsupported DataFrame type"),
    ):
        from_dataframe(df_pd, {"media_id": "track:1"})


def test_invalid_backend():
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_type="beat",
        extent_type="time",
        time=[0.1],
        value=[1],
    )
    with pytest.raises(BoppArgumentError, match="Unsupported backend"):
        to_dataframe(ann, backend="invalid_backend")  # type: ignore[arg-type]


def test_from_dataframe_unsupported_type():
    # Verify rejection when passed a non-DataFrame object
    with pytest.raises(BoppArgumentError, match="Unsupported DataFrame type"):
        from_dataframe({"not": "a_dataframe"}, {"media_id": "1"})


def test_from_dataframe_no_header():
    df = pd.DataFrame({"extent:time:time": [0.1], "payload:beat:value": [1]})
    header = None

    with pytest.raises(
        BoppValidationError, match="Missing or invalid required 'media_id'"
    ):
        from_dataframe(df, header=header)


type NonNegative = Annotated[float, msgspec.Meta(ge=0.0)]
type NonNegativeInt = Annotated[int, msgspec.Meta(ge=0)]
type IntOrFloat = NonNegativeInt | NonNegative


class Controller(enum.StrEnum):
    SUSTAIN = "sustain"
    SOFT = "soft"


@pytest.mark.parametrize(
    ("field_type", "expected"),
    [
        (list[int], pl.Int64),
        (list[float], pl.Float64),
        (list[bool], pl.Boolean),
        (list[str], pl.String),
        (list[int | float], pl.Float64),
        # constrained items are PEP 695 aliases in the generated models
        (Annotated[list[NonNegative], msgspec.Meta(description="d")], pl.Float64),
        (list[IntOrFloat], pl.Float64),
        # nullable items and optional columns
        (list[int | None], pl.Int64),
        (list[bool | None], pl.Boolean),
        (list[NonNegative | None] | msgspec.UnsetType, pl.Float64),
        (Annotated[list[str | None], msgspec.Meta(description="d")] | None, pl.String),
        (list[Literal["a", "b"]], pl.String),
        (list[Controller], pl.String),
        # no single scalar dtype: left to polars inference
        (list[str | float], None),
        (list[Any], None),
        (list[list[int]], None),
        (list[dict[str, Any]], None),
        (str, None),
    ],
)
def test_polars_column_dtype(field_type, expected):
    assert _polars_column_dtype(field_type, pl) == expected


def test_polars_column_dtypes_match_csv_reader():
    """Every registered column is read from CSV with the dtype it is written with."""
    registry = get_registry("1.0")
    complex_registry = registry["COMPLEX_FIELDS_REGISTRY"]
    checked = 0
    for facet, registry_key in FACET_REGISTRY_KEYS.items():
        tag_field = f"{facet}_type"
        for tag, struct_cls in registry[registry_key].items():
            for field in msgspec.structs.fields(struct_cls):
                written = _polars_column_dtype(field.type, pl)
                read, is_complex = _type_to_polars_dtype(
                    field.type, field.name, tag, tag_field, complex_registry, pl
                )
                if written is None or is_complex:
                    continue
                assert read == written, f"{facet}:{tag}:{field.name}"
                checked += 1
    assert checked > 0


@pytest.mark.parametrize(
    "column",
    [
        "extent:time:time",
        "extent:time_interval:duration",
        "extent:midi_interval:duration",
        "extent:time_frequency_box:freq_min",
        "extent:pixel_box:width",
        "payload:tempo:value",
        "payload:frequency_hz:value",
        "confidence:likelihood:confidence",
    ],
)
def test_polars_column_dtype_constrained_numbers(column):
    """Number columns with constrained (aliased) items map to Float64."""
    facet, tag, field_name = column.split(":")
    struct_cls = get_registry("1.0")[FACET_REGISTRY_KEYS[facet]][tag]
    field = next(f for f in msgspec.structs.fields(struct_cls) if f.name == field_name)
    assert _polars_column_dtype(field.type, pl) == pl.Float64


def test_validate_dataframe_columns_grouping_indices():
    # Non-canonical integer index (leading zero)
    with pytest.raises(
        BoppValidationError, match="Invalid grouping column format 'group.00.ids'"
    ):
        _validate_dataframe_columns(["payload:beat:value", "group.00.ids"], "1.0")

    # Negative index
    with pytest.raises(
        BoppValidationError, match="Invalid grouping column format 'group.-1.ids'"
    ):
        _validate_dataframe_columns(["payload:beat:value", "group.-1.ids"], "1.0")

    # Non-contiguous indices (missing 1)
    with pytest.raises(
        BoppValidationError,
        match=r"Grouping column indices must be contiguous starting from 0 \(expected \[0, 1\], got \[0, 2\]\)",
    ):
        _validate_dataframe_columns(
            ["payload:beat:value", "group.0.ids", "group.2.ids"], "1.0"
        )


def test_to_and_from_dataframe_marker_payload():
    ann = create(
        media_id="track:marker_df",
        bopp_version="1.0",
        payload_type="marker",
        extent_type="time",
        time=[0.1, 0.5, 1.2],
        description="cues",
        sandbox={"extra": "data"},
    )
    for backend in ("pandas", "pandas-pyarrow", "polars"):
        ann_df = to_dataframe(ann, backend=backend)
        assert ann_df.header["media_id"] == "track:marker_df"
        assert ann_df.header["payload"]["payload_type"] == "marker"
        assert ann_df.header["payload"]["description"] == "cues"
        # No payload columns in the DataFrame
        assert not any(col.startswith("payload:") for col in ann_df.df.columns)
        assert "extent:time:time" in ann_df.df.columns

        reconstructed = from_dataframe(ann_df)
        assert reconstructed.media_id == ann.media_id
        assert isinstance(reconstructed.payload, MarkerPayload)
        assert reconstructed.payload.description == "cues"
        assert reconstructed.extent.time == [0.1, 0.5, 1.2]
        assert reconstructed.id == ann.id


def test_to_and_from_dataframe_marker_payload_without_extent():
    ann = create(
        media_id="track:marker_global",
        bopp_version="1.0",
        payload_type="marker",
        description="global_marker",
    )
    for backend in ("pandas", "pandas-pyarrow", "polars"):
        ann_df = to_dataframe(ann, backend=backend)
        assert ann_df.header["media_id"] == "track:marker_global"
        assert ann_df.header["payload"]["payload_type"] == "marker"
        assert ann_df.header["payload"]["description"] == "global_marker"
        assert len(ann_df.df.columns) == 0

        reconstructed = from_dataframe(ann_df)
        assert reconstructed.media_id == ann.media_id
        assert isinstance(reconstructed.payload, MarkerPayload)
        assert reconstructed.payload.description == "global_marker"
        assert reconstructed.extent is msgspec.UNSET
        assert reconstructed.id == ann.id


def test_from_dataframe_facet_header_without_type_tag():
    df = pd.DataFrame({"extent:time:time": [0.1], "payload:beat:value": [1]})
    header = {
        "bopp_version": "1.0",
        "media_id": "track:no_tag",
        "confidence": {"custom_note": "no_tag_present"},
    }
    ann = from_dataframe(df, header=header)
    assert ann.confidence is msgspec.UNSET


def test_to_and_from_dataframe_lyrics_payload():
    ann = create(
        media_id="track:lyrics_df",
        bopp_version="1.0",
        payload_type="lyrics",
        extent_type="time",
        time=[0.1, 0.5],
        value=["hello", "world"],
        language="en",
    )
    for backend in ("pandas", "polars"):
        ann_df = to_dataframe(ann, backend=backend)
        assert ann_df.header["payload"]["language"] == "en"
        assert "payload:lyrics:value" in ann_df.df.columns

        reconstructed = from_dataframe(ann_df)
        assert reconstructed.payload.language == "en"
        assert reconstructed.payload.value == ["hello", "world"]


def test_to_and_from_dataframe_frequency_hz_with_voiced():
    ann = create(
        media_id="track:freq_voiced_df",
        bopp_version="1.0",
        payload_type="frequency_hz",
        extent_type="time",
        time=[0.0, 0.5],
        value=[440.0, 880.0],
        voiced=[True, False],
    )
    for backend in ("pandas", "polars"):
        ann_df = to_dataframe(ann, backend=backend)
        assert "payload:frequency_hz:value" in ann_df.df.columns
        assert "payload:frequency_hz:voiced" in ann_df.df.columns

        reconstructed = from_dataframe(ann_df)
        assert reconstructed.payload.value == [440.0, 880.0]
        assert reconstructed.payload.voiced == [True, False]


def test_to_and_from_dataframe_object_with_description():
    ann = create(
        media_id="track:object_desc_df",
        bopp_version="1.0",
        payload_type="object",
        extent_type="time",
        time=[0.1, 0.5],
        value=[{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        description="test objects",
    )
    for backend in ("pandas", "polars"):
        ann_df = to_dataframe(ann, backend=backend)
        assert ann_df.header["payload"]["description"] == "test objects"
        assert "payload:object:value" in ann_df.df.columns

        reconstructed = from_dataframe(ann_df)
        assert reconstructed.payload.description == "test objects"
        assert reconstructed.payload.value == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_to_and_from_dataframe_note_payload():
    ann = create(
        media_id="track:note_spn_df",
        bopp_version="1.0",
        payload_type="note",
        extent_type="time",
        time=[0.0, 1.0],
        value=["C4", "C♯4"],
    )
    for backend in ("pandas", "polars"):
        ann_df = to_dataframe(ann, backend=backend)
        assert "payload:note:value" in ann_df.df.columns

        reconstructed = from_dataframe(ann_df)
        assert reconstructed.payload.value == ["C4", "C♯4"]
