import enum
from typing import Annotated, Any, Literal
from unittest.mock import patch

import msgspec
import pandas as pd
import polars as pl
import pytest

from bopp.core import BoppArgumentError, BoppIOError, BoppValidationError, create
from bopp.io import _type_to_polars_dtype
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.confidence.likelihood import LikelihoodConfidence
from bopp.models.v1.extent.score_interval import ScoreInterval
from bopp.models.v1.extent.times import Times
from bopp.models.v1.payload.object import ObjectPayload
from bopp.models.v1.payload.onset import OnsetPayload
from bopp.registries import FACET_REGISTRY_KEYS, get_registry
from bopp.util import (
    _extract_facet_data,
    _get_tag,
    _polars_column_dtype,
    _validate_dataframe_columns,
    extract_header,
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


def test_extract_header_with_scalar_facet_attributes():
    """Verify extract_header includes scalar fields on facets when present."""
    class ScalarPayload(msgspec.Struct, tag_field="payload_type", tag="onset"):
        value: list[int]
        unit: str = "custom_unit"

    class ScalarExtent(msgspec.Struct, tag_field="extent_type", tag="time"):
        time: list[float]
        time_unit: str = "seconds"

    # Annotation with facet as a non-struct object to ensure non-struct facets are ignored
    class NonStructExtentAnnotation(msgspec.Struct):
        media_id: str
        extent: Any = "not_a_struct"
        payload: ScalarPayload | None = None

    ann_non_struct = NonStructExtentAnnotation(
        media_id="track:non_struct",
        payload=ScalarPayload(value=[1, 2], unit="custom_unit"),
    )
    header_non_struct = extract_header(ann_non_struct)  # type: ignore[arg-type]
    assert "extent" not in header_non_struct

    ann = Annotation(
        media_id="track:scalar_test",
        bopp_version="1.0",
        extent=ScalarExtent(time=[0.1, 0.2], time_unit="seconds"),  # type: ignore[arg-type]
        payload=ScalarPayload(value=[1, 2], unit="custom_unit"),  # type: ignore[arg-type]
    )

    header = extract_header(ann)
    assert "payload" in header
    assert header["payload"]["payload_type"] == "onset"
    assert header["payload"]["unit"] == "custom_unit"
    assert "extent" in header
    assert header["extent"]["extent_type"] == "time"
    assert header["extent"]["time_unit"] == "seconds"


def test_extract_facet_data_scalar_attributes():
    class DummyFacet(msgspec.Struct, tag_field="payload_type", tag="onset"):
        value: list[int]
        sample_rate: float

    facet = DummyFacet(value=[1, 2], sample_rate=44100.0)
    target_dict: dict[str, list[int]] = {}
    attrs_dict: dict[str, dict[str, float]] = {}

    _extract_facet_data(facet, "payload", target_dict, attrs_dict)

    assert target_dict == {"payload:onset:value": [1, 2]}
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
    df_raw = to_dataframe(ann, backend="pandas", encode_complex=False)
    assert df_raw["payload:object:value"][0] == {"k": "v"}
    assert df_raw["extent:score_interval:quarter"][0] == [0, 1]

    # With encode_complex=True, columns contain JSON-encoded strings
    df_encoded = to_dataframe(ann, backend="pandas", encode_complex=True)
    assert df_encoded["payload:object:value"][0] == '{"k":"v"}'
    assert df_encoded["extent:score_interval:quarter"][0] == "[0,1]"
    assert df_encoded["extent:score_interval:duration"][0] == "[1,2]"


def test_validate_dataframe_columns_invalid_format():
    """Verify that DataFrame column header validation rejects column names that do not
    adhere to the required 'facet:type:field' 3-part naming format.
    """
    invalid_columns = [
        "invalid_column_no_colons",
        "extent:time",  # Too few parts (2)
        "payload:onset:value:extra",  # Too many parts (4)
        ":::",  # Empty parts
    ]
    for col in invalid_columns:
        with pytest.raises(
            BoppValidationError, match="Expected exactly 3 colon-separated fields"
        ):
            _validate_dataframe_columns([col], "1.0")


def test_validate_dataframe_columns_rules():
    """Verify DataFrame column validation rules 2, 3, and 4."""
    # Rule 2: Mixed types for facet
    with pytest.raises(BoppValidationError, match="Mixed types for facet 'payload'"):
        _validate_dataframe_columns(["payload:onset:value", "payload:beat:value"], "1.0")

    # Rule 3a: Invalid facet
    with pytest.raises(BoppValidationError, match="Invalid facet 'unknown_facet'"):
        _validate_dataframe_columns(["unknown_facet:onset:value"], "1.0")

    # Rule 3b: Unrecognized type tag in registry
    with pytest.raises(BoppValidationError, match="Unrecognized type 'unknown_tag'"):
        _validate_dataframe_columns(["payload:unknown_tag:value"], "1.0")

    # Rule 4: Field does not exist on target type struct
    with pytest.raises(
        BoppValidationError, match="Field 'nonexistent_field' in column .* does not exist"
    ):
        _validate_dataframe_columns(["payload:onset:nonexistent_field"], "1.0")


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


def test_to_dataframe_filters_tag_discriminators_and_unset_metadata():
    """Verify that `to_dataframe` correctly extracts array data and metadata headers while
    omitting tag discriminator fields (e.g. `payload_type`) and UNSET optional fields.
    """
    ann = Annotation(
        media_id="track:skip_test",
        bopp_version="1.0",
        metadata=msgspec.UNSET,
        extent=Times(time=[0.1, 0.2]),
        payload=OnsetPayload(value=[1, 1]),
        confidence=LikelihoodConfidence(confidence=[0.8, 0.9]),
    )

    df = to_dataframe(ann, backend="pandas")

    # Verify parallel array data columns are created correctly
    assert list(df["extent:time:time"]) == [0.1, 0.2]
    assert list(df["payload:onset:value"]) == [1, 1]
    assert list(df["confidence:likelihood:confidence"]) == [0.8, 0.9]

    # Verify discriminator tag fields are excluded from DataFrame columns
    assert "extent:time:extent_type" not in df.columns
    assert "payload:onset:payload_type" not in df.columns
    assert "confidence:likelihood:confidence_type" not in df.columns

    # Verify UNSET metadata is not attached to DataFrame attributes
    assert "metadata" not in df.attrs


def test_to_dataframe_covers_extent_and_confidence_branches():
    """Verify that to_dataframe extracts parallel arrays and attributes when both extent and confidence exist."""
    ann = create(
        bopp_version="1.0",
        media_id="track:facets_coverage",
        payload_kind="onset",
        extent_kind="time",
        confidence_kind="likelihood",
        time=[0.1, 0.2],
        value=[1, 1],
        confidence=[0.85, 0.95],
    )
    df = to_dataframe(ann, backend="pandas")
    assert "extent:time:time" in df.columns
    assert "confidence:likelihood:confidence" in df.columns


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


def test_from_dataframe_with_scalar_facet_attributes():
    """Verify from_dataframe accurately restores scalar facet attributes from df.attrs across extent, payload, confidence."""
    class ExtentWithScalar(msgspec.Struct, tag_field="extent_type", tag="time"):
        time: list[float]
        origin: str = "start"

    class PayloadWithScalar(msgspec.Struct, tag_field="payload_type", tag="onset"):
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
        "PAYLOAD_TYPE_REGISTRY": {"onset": PayloadWithScalar},
        "CONFIDENCE_TYPE_REGISTRY": {"likelihood": ConfWithScalar},
        "COMPLEX_FIELDS_REGISTRY": {},
    }

    with patch("bopp.util.get_registry", return_value=mock_registry):
        df = pd.DataFrame({
            "extent:time:time": [0.1, 0.2],
            "payload:onset:value": [1, 2],
            "confidence:likelihood:confidence": [0.9, 0.95],
        })
        df.attrs = {
            "bopp_version": "1.0",
            "media_id": "track:with_scalars",
            "extent": {"origin": "custom_origin"},
            "payload": {"source": "custom_source"},
            "confidence": {"method": "custom_method"},
        }

        reconstructed = from_dataframe(df)
        assert getattr(reconstructed.extent, "origin") == "custom_origin"
        assert getattr(reconstructed.payload, "source") == "custom_source"
        assert getattr(reconstructed.confidence, "method") == "custom_method"


def test_from_dataframe_missing_required_payload_facet():
    """Verify that reconstructing an Annotation from a DataFrame fails validation
    if the DataFrame lacks mandatory 'payload:*' columns required by the schema.
    """
    df = pd.DataFrame({"extent:time:time": [0.1, 0.2]})
    df.attrs = {
        "bopp_version": "1.0",
        "media_id": "track:no_payload",
    }

    with pytest.raises(msgspec.ValidationError, match="Object missing required field `payload`"):
        from_dataframe(df)


def test_from_dataframe_missing_media_id():
    df = pd.DataFrame({"payload:onset:value": [1, 2]})
    df.attrs = {
        "bopp_version": "1.0",
    }

    with pytest.raises(BoppValidationError, match="Missing or invalid required 'media_id'"):
        from_dataframe(df)


def test_from_dataframe_extra_attrs():
    # Test metadata and annotated_domain attributes handling in from_dataframe
    df = pd.DataFrame({"extent:time:time": [0.1], "payload:onset:value": [1]})
    df.attrs = {
        "bopp_version": "1.0",
        "media_id": "track:extra_attrs",
        "metadata": {"metadata_type": "other", "notes": "nonsense"},
    }

    reconstructed = from_dataframe(df)
    assert reconstructed.media_id == "track:extra_attrs"


def test_to_dataframe_raises_ioerror_when_backend_dependency_missing():
    """Verify that `to_dataframe` raises an actionable BoppIOError when attempting
    to use a DataFrame backend whose underlying library is not installed.
    """
    ann = create(
        bopp_version="1.0",
        media_id="track:1",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
    )

    with patch.dict("sys.modules", {"pandas": None}):
        with pytest.raises(BoppIOError, match="pandas is required"):
            to_dataframe(ann, backend="pandas")

    with patch.dict("sys.modules", {"pyarrow": None}):
        with pytest.raises(BoppIOError, match="pyarrow is required"):
            to_dataframe(ann, backend="pandas-pyarrow")


def test_from_dataframe_raises_argument_error_when_dataframe_libraries_missing():
    """Verify that `from_dataframe` raises a BoppArgumentError when required DataFrame
    libraries (pandas/polars) cannot be imported to inspect the input object.
    """
    df_pd = pd.DataFrame({"payload:onset:value": [1]})
    with patch.dict("sys.modules", {"pandas": None}):
        with pytest.raises(BoppArgumentError, match="Unsupported DataFrame type"):
            from_dataframe(df_pd)

    df_pl = pl.DataFrame({"payload:onset:value": [1]})
    with patch.dict("sys.modules", {"polars": None}):
        with pytest.raises(BoppArgumentError, match="Unsupported DataFrame type"):
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
    # Verify rejection when passed a non-DataFrame object
    with pytest.raises(BoppArgumentError, match="Unsupported DataFrame type"):
        from_dataframe({"not": "a_dataframe"})

    # Verify fallback handling when polars is not installed and input is not a pandas DataFrame
    with patch.dict("sys.modules", {"polars": None}):
        with pytest.raises(BoppArgumentError, match="Unsupported DataFrame type"):
            from_dataframe({"not": "a_dataframe"})


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
    string_registry = registry["STRING_FIELDS_REGISTRY"]
    checked = 0
    for facet, registry_key in FACET_REGISTRY_KEYS.items():
        tag_field = f"{facet}_type"
        for tag, struct_cls in registry[registry_key].items():
            for field in msgspec.structs.fields(struct_cls):
                written = _polars_column_dtype(field.type, pl)
                read, is_complex = _type_to_polars_dtype(
                    field.type, field.name, tag, tag_field, complex_registry, string_registry, pl
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
        "payload:note_hz:value",
        "confidence:likelihood:confidence",
    ],
)
def test_polars_column_dtype_constrained_numbers(column):
    """Number columns with constrained (aliased) items map to Float64."""
    facet, tag, field_name = column.split(":")
    struct_cls = get_registry("1.0")[FACET_REGISTRY_KEYS[facet]][tag]
    field = next(f for f in msgspec.structs.fields(struct_cls) if f.name == field_name)
    assert _polars_column_dtype(field.type, pl) == pl.Float64
