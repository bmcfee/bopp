import base64
import uuid
import warnings
from typing import Any

import msgspec
import pytest

from bopp.base import BoppBase, GroupingBase
from bopp.core import (
    _extract_kwargs,
    create,
    ensure_annotation_id,
    validate,
    validate_annotation_id,
)
from bopp.exceptions import (
    BoppArgumentError,
    BoppArrayError,
    BoppRegistryError,
    BoppValidationError,
)
from bopp.models.v1.annotation import Annotation
from bopp.models.v1.grouping import Grouping
from bopp.models.v1.metadata.human import HumanAnnotationMetadata
from bopp.models.v1.payload.onset import OnsetPayload


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
        bopp_version="1.0",
        media_id="audio:123",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.5],
        value=[1, 1],
    )
    assert isinstance(ann, Annotation)
    assert ann.media_id == "audio:123"
    assert ann.payload.__struct_config__.tag == "onset"
    assert getattr(ann, "id", None) is not None
    val = uuid.UUID(ann.id)
    assert val.version == 5


def test_create_generate_id_false():
    ann = create(
        bopp_version="1.0",
        media_id="audio:123",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.5],
        value=[1, 1],
        generate_id=False,
    )
    assert getattr(ann, "id", msgspec.UNSET) is msgspec.UNSET


def test_create_full():
    ann = create(
        bopp_version="1.0",
        media_id="audio:456",
        payload_kind="onset",
        extent_kind="time",
        confidence_kind="likelihood",
        time=[0.1, 0.5],
        value=[1, 1],
        confidence=[0.9, 0.95],
        sandbox={"custom_key": "custom_value", "notes": [1, 2, 3]},
    )
    assert ann.extent is not None
    assert ann.extent.__struct_config__.tag == "time"
    assert ann.confidence is not None
    assert ann.confidence.__struct_config__.tag == "likelihood"
    assert getattr(ann, "sandbox", None) == {"custom_key": "custom_value", "notes": [1, 2, 3]}


def test_uuid5_idempotence_and_mutation():
    ann = create(
        bopp_version="1.0",
        media_id="audio:123",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.5],
        value=[1, 1],
    )
    initial_id = ann.id

    # Verify idempotency
    assert ensure_annotation_id(ann) == initial_id
    assert validate_annotation_id(ann) is True

    # Mutate payload and verify failure when validate_annotation_id checks the existing ID
    ann.payload.value = [1, 2]
    with pytest.raises(BoppValidationError, match="Annotation ID mismatch"):
        validate_annotation_id(ann)


def test_validate_annotation_id_missing():
    ann = create(
        bopp_version="1.0",
        media_id="audio:123",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.5],
        value=[1, 1],
        generate_id=False,
    )
    assert validate_annotation_id(ann) is False


def test_create_with_parents():
    parent_id = str(uuid.uuid4())
    ann = create(
        bopp_version="1.0",
        media_id="audio:123",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
        parents=[parent_id],
    )
    assert getattr(ann, "parents", None) == [parent_id]


def test_create_invalid_payload_kind():
    with pytest.raises(BoppRegistryError, match="Unrecognized kind identifier"):
        create(bopp_version="1.0", media_id="audio:123", payload_kind="non_existent_kind")


def test_create_invalid_extent_kind():
    with pytest.raises(BoppRegistryError, match="Unrecognized extent kind"):
        create(
            bopp_version="1.0",
            media_id="audio:123",
            payload_kind="onset",
            extent_kind="non_existent_extent",
            time=[0.1],
            value=[1],
        )


def test_create_invalid_confidence_kind():
    with pytest.raises(BoppRegistryError, match="Unrecognized confidence kind"):
        create(
            bopp_version="1.0",
            media_id="audio:123",
            payload_kind="onset",
            confidence_kind="non_existent_conf",
            time=[0.1],
            value=[1],
        )


def test_create_unconsumed_kwargs():
    with pytest.raises(BoppArgumentError, match="Unconsumed keyword arguments"):
        create(
            bopp_version="1.0",
            media_id="audio:123",
            payload_kind="onset",
            time=[0.1],
            value=[1],
            unused_param="invalid",
        )


def test_create_mismatched_array_lengths():
    with pytest.raises(BoppArrayError, match="Length mismatch"):
        create(
            bopp_version="1.0",
            media_id="audio:123",
            payload_kind="onset",
            extent_kind="time",
            time=[0.1, 0.5],
            value=[1],
        )


def test_validate_struct():
    ann = create(
        bopp_version="1.0",
        media_id="audio:123",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
        sandbox={"experiment_id": "exp-42"},
    )
    assert validate(ann) is True


def test_validate_dict():
    data = {
        "bopp_version": "1.0",
        "media_id": "audio:123",
        "payload": {"payload_type": "onset", "time": [0.1], "value": [1]},
        "sandbox": {"experiment_id": "exp-42"},
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


def test_repr_mimebundle():
    ann = create(
        bopp_version="1.0",
        media_id="audio:repr_test",
        payload_kind="onset",
        extent_kind="time",
        time=[0.25, 0.75],
        value=[1, 1],
    )

    bundle = ann._repr_mimebundle_()
    assert isinstance(bundle, dict)
    mime_type = "application/vnd.bopp+msgpack"
    assert mime_type in bundle

    encoded_str = bundle[mime_type]
    assert isinstance(encoded_str, str)

    raw_bytes = base64.b64decode(encoded_str.encode("ascii"))
    decoded_ann = msgspec.msgpack.decode(raw_bytes, type=Annotation)
    assert decoded_ann == ann

    # Check that invocation with include and exclude succeeds
    bundle_filtered = ann._repr_mimebundle_(include=["text/plain"], exclude=[mime_type])
    assert mime_type in bundle_filtered


def test_grouping_without_labels():
    grouping = Grouping(name="sections", ids=[0, 1, 0, 1, 2])
    assert grouping.name == "sections"
    assert grouping.ids == [0, 1, 0, 1, 2]
    assert grouping.labels is msgspec.UNSET


def test_grouping_valid_bijective_labels():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        grouping = Grouping(
            name="sections",
            ids=[0, 1, 0, 2],
            labels={"verse": 0, "chorus": 1, "bridge": 2},
        )
    assert grouping.labels == {"verse": 0, "chorus": 1, "bridge": 2}


def test_grouping_negative_ids_allowed_unmapped():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        grouping = Grouping(
            name="sections",
            ids=[-1, 0, -1, 1],
            labels={"verse": 0, "chorus": 1},
        )
    assert grouping.ids == [-1, 0, -1, 1]


def test_grouping_warns_unmapped_non_negative_ids():
    with pytest.warns(UserWarning, match="contains non-negative group IDs not included in label mapping: \\[2\\]"):
        Grouping(
            name="sections",
            ids=[0, 1, 2],
            labels={"verse": 0, "chorus": 1},
        )


def test_grouping_duplicate_id_values_raises():
    with pytest.raises(
        BoppValidationError,
        match="Grouping assigns two or more labels to the same group identifier",
    ):
        Grouping(
            name="sections",
            ids=[0, 0],
            labels={"verse": 0, "chorus": 0},
        )


def test_grouping_non_list_ids_ignored_in_root_lengths():
    class CustomGrouping(msgspec.Struct):
        name: str
        ids: Any = None

    class CustomAnnotation(BoppBase):
        media_id: str
        payload: OnsetPayload
        groupings: list[Any]

    ann = CustomAnnotation(
        media_id="audio:1",
        payload=OnsetPayload(value=[1, 2]),
        groupings=[CustomGrouping(name="test", ids=None)],
    )
    assert ann.media_id == "audio:1"


def test_grouping_with_labels_and_none_or_unset_ids():
    class GroupingWithoutIds(GroupingBase):
        name: str
        labels: dict[str, int]
        ids: Any = None

    g_none = GroupingWithoutIds(name="test", labels={"a": 0}, ids=None)
    assert g_none.labels == {"a": 0}

    g_unset = GroupingWithoutIds(name="test", labels={"a": 0}, ids=msgspec.UNSET)
    assert g_unset.labels == {"a": 0}


def test_duplicate_grouping_names_raises():
    with pytest.raises(
        BoppValidationError,
        match="Annotation contains multiple groupings with the same name: 'sections'",
    ):
        Annotation(
            media_id="audio:1",
            bopp_version="1.0.0",
            metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
            payload=OnsetPayload(value=[1, 1]),
            groupings=[
                Grouping(name="sections", ids=[0, 1]),
                Grouping(name="sections", ids=[1, 0]),
            ],
        )


def test_create_with_metadata_and_groupings():
    # 1. metadata_kind + kwargs
    ann1 = create(
        media_id="audio:1",
        payload_kind="onset",
        metadata_kind="human",
        annotator_id="user_1",
        tool="sonic-visualiser",
        value=[1, 1],
        groupings=[Grouping(name="beats", ids=[0, 1])],
    )
    assert isinstance(ann1.metadata, HumanAnnotationMetadata)
    assert ann1.metadata.annotator_id == "user_1"
    assert ann1.metadata.tool == "sonic-visualiser"
    assert len(ann1.groupings) == 1
    assert ann1.groupings[0].name == "beats"

    # 2. metadata as dict + groupings as list of dicts
    ann2 = create(
        media_id="audio:1",
        payload_kind="onset",
        metadata={"metadata_type": "human", "annotator_id": "user_2", "tool": "audacity"},
        groupings=[{"name": "sections", "ids": [0, 1], "labels": {"a": 0, "b": 1}}],
        value=[1, 1],
    )
    assert isinstance(ann2.metadata, HumanAnnotationMetadata)
    assert ann2.metadata.annotator_id == "user_2"
    assert isinstance(ann2.groupings[0], Grouping)
    assert ann2.groupings[0].name == "sections"

    # 3. metadata as struct instance
    ann3 = create(
        media_id="audio:1",
        payload_kind="onset",
        metadata=HumanAnnotationMetadata(annotator_id="user_3", tool="manual"),
        value=[1, 1],
    )
    assert ann3.metadata.annotator_id == "user_3"


def test_create_metadata_errors():
    with pytest.raises(BoppArgumentError, match="Cannot specify both 'metadata' and 'metadata_kind'"):
        create(
            media_id="audio:1",
            payload_kind="onset",
            metadata_kind="human",
            metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
            value=[1],
        )

    with pytest.raises(BoppRegistryError, match="Unrecognized metadata kind: 'nonexistent'"):
        create(
            media_id="audio:1",
            payload_kind="onset",
            metadata_kind="nonexistent",
            value=[1],
        )

    with pytest.raises(BoppRegistryError, match="Unrecognized metadata kind: 'bad_type'"):
        create(
            media_id="audio:1",
            payload_kind="onset",
            metadata={"metadata_type": "bad_type"},
            value=[1],
        )

