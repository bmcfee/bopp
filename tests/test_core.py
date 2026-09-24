import msgspec
import pytest

from bopp.core import (
    BoppArgumentError,
    BoppArrayLengthMismatchError,
    BoppError,
    BoppRegistryError,
    BoppValidationError,
    _extract_kwargs,
    create,
    validate,
)
from bopp.models.v1.annotation import Annotation


class DummyStruct(msgspec.Struct):
    foo: str
    bar: int = 0


def test_extract_kwargs():
    kwargs = {"foo": "test", "bar": 10, "extra": "ignore"}
    extracted = _extract_kwargs(DummyStruct, kwargs)
    assert extracted == {"foo": "test", "bar": 10}
    assert kwargs == {"extra": "ignore"}


def test_create_minimal():
    ann = create(
        media_id="audio:123",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.5],
        value=[1, 1],
    )
    assert isinstance(ann, Annotation)
    assert ann.media_id == "audio:123"
    assert ann.payload.__struct_config__.tag == "onset"


def test_create_full():
    ann = create(
        media_id="audio:456",
        payload_kind="onset",
        extent_kind="time",
        confidence_kind="likelihood",
        time=[0.1, 0.5],
        value=[1, 1],
        confidence=[0.9, 0.95],
    )
    assert ann.extent is not None
    assert ann.extent.__struct_config__.tag == "time"
    assert ann.confidence is not None
    assert ann.confidence.__struct_config__.tag == "likelihood"


def test_create_invalid_payload_kind():
    with pytest.raises(BoppRegistryError, match="Unrecognized kind identifier"):
        create(media_id="audio:123", payload_kind="non_existent_kind")


def test_create_invalid_extent_kind():
    with pytest.raises(BoppRegistryError, match="Unrecognized extent kind"):
        create(
            media_id="audio:123",
            payload_kind="onset",
            extent_kind="non_existent_extent",
            time=[0.1],
            value=[1],
        )


def test_create_invalid_confidence_kind():
    with pytest.raises(BoppRegistryError, match="Unrecognized confidence kind"):
        create(
            media_id="audio:123",
            payload_kind="onset",
            confidence_kind="non_existent_conf",
            time=[0.1],
            value=[1],
        )


def test_create_unconsumed_kwargs():
    with pytest.raises(BoppArgumentError, match="Unconsumed keyword arguments"):
        create(
            media_id="audio:123",
            payload_kind="onset",
            time=[0.1],
            value=[1],
            unused_param="invalid",
        )


def test_create_mismatched_array_lengths():
    with pytest.raises(BoppArrayLengthMismatchError, match="Length mismatch"):
        create(
            media_id="audio:123",
            payload_kind="onset",
            extent_kind="time",
            time=[0.1, 0.5],
            value=[1],
        )


def test_validate_struct():
    ann = create(
        media_id="audio:123",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
    )
    assert validate(ann) is True


def test_validate_dict():
    data = {
        "bopp_version": "1.0.0",
        "media_id": "audio:123",
        "payload": {"payload_type": "onset", "time": [0.1], "value": [1]},
    }
    assert validate(data, target_type=Annotation) is True


def test_validate_missing_target_type():
    with pytest.raises(BoppArgumentError, match="target_type must be provided"):
        validate({"media_id": "123"})


def test_validate_invalid_data():
    invalid_data = {
        "bopp_version": "v1",
        "media_id": "audio:123",
        "payload": {"payload_type": "onset", "time": "not_a_list", "value": [1]},
    }
    with pytest.raises(BoppValidationError, match="Validation failed"):
        validate(invalid_data, target_type=Annotation)
