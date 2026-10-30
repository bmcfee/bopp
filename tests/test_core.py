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
from bopp.models.v1.payload.beat import BeatPositionPayload
from bopp.models.v1.payload.marker import MarkerPayload


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
        payload_type="marker",
        extent_type="time",
        time=[0.1, 0.5],
        description="onset",
    )
    assert isinstance(ann, Annotation)
    assert ann.media_id == "audio:123"
    assert ann.payload.__struct_config__.tag == "marker"
    assert ann.payload.description == "onset"
    assert getattr(ann, "id", None) is not None
    val = uuid.UUID(ann.id)
    assert val.version == 5


def test_create_generate_id_false():
    ann = create(
        bopp_version="1.0",
        media_id="audio:123",
        payload_type="marker",
        extent_type="time",
        time=[0.1, 0.5],
        description="onset",
        generate_id=False,
    )
    assert getattr(ann, "id", msgspec.UNSET) is msgspec.UNSET


def test_create_full():
    ann = create(
        bopp_version="1.0",
        media_id="audio:456",
        payload_type="marker",
        extent_type="time",
        confidence_type="likelihood",
        time=[0.1, 0.5],
        description="onset",
        confidence=[0.9, 0.95],
        sandbox={"custom_key": "custom_value", "notes": [1, 2, 3]},
    )
    assert ann.extent is not None
    assert ann.extent.__struct_config__.tag == "time"
    assert ann.confidence is not None
    assert ann.confidence.__struct_config__.tag == "likelihood"
    assert getattr(ann, "sandbox", None) == {
        "custom_key": "custom_value",
        "notes": [1, 2, 3],
    }


def test_uuid5_idempotence_and_mutation():
    ann = create(
        bopp_version="1.0",
        media_id="audio:123",
        payload_type="marker",
        extent_type="time",
        time=[0.1, 0.5],
        description="onset",
    )
    initial_id = ann.id

    # Verify idempotency
    assert ensure_annotation_id(ann) == initial_id
    assert validate_annotation_id(ann) is True

    # Mutate payload and verify failure when validate_annotation_id checks the existing ID
    ann.payload.description = "modified_onset"
    with pytest.raises(BoppValidationError, match="Annotation ID mismatch"):
        validate_annotation_id(ann)


def test_validate_annotation_id_missing():
    ann = create(
        bopp_version="1.0",
        media_id="audio:123",
        payload_type="marker",
        extent_type="time",
        time=[0.1, 0.5],
        description="onset",
        generate_id=False,
    )
    assert validate_annotation_id(ann) is False


def test_create_with_parents():
    parent_id = str(uuid.uuid4())
    ann = create(
        bopp_version="1.0",
        media_id="audio:123",
        payload_type="marker",
        extent_type="time",
        time=[0.1],
        description="onset",
        parents=[parent_id],
    )
    assert getattr(ann, "parents", None) == [parent_id]


def test_create_invalid_payload_type():
    with pytest.raises(BoppRegistryError, match="Unrecognized payload type"):
        create(
            bopp_version="1.0", media_id="audio:123", payload_type="non_existent_type"
        )


def test_create_invalid_extent_type():
    with pytest.raises(BoppRegistryError, match="Unrecognized extent type"):
        create(
            bopp_version="1.0",
            media_id="audio:123",
            payload_type="marker",
            extent_type="non_existent_extent",
            time=[0.1],
            description="onset",
        )


def test_create_invalid_confidence_type():
    with pytest.raises(BoppRegistryError, match="Unrecognized confidence type"):
        create(
            bopp_version="1.0",
            media_id="audio:123",
            payload_type="marker",
            confidence_type="non_existent_conf",
            description="onset",
        )


def test_create_unconsumed_kwargs():
    with pytest.raises(BoppArgumentError, match="Unconsumed keyword arguments"):
        create(
            bopp_version="1.0",
            media_id="audio:123",
            payload_type="marker",
            description="onset",
            unused_param="invalid",
        )


def test_create_mismatched_array_lengths():
    with pytest.raises(BoppArrayError, match="Length mismatch"):
        create(
            bopp_version="1.0",
            media_id="audio:123",
            payload_type="beat",
            extent_type="time",
            time=[0.1, 0.5],
            value=[1],
        )


def test_validate_struct():
    ann = create(
        bopp_version="1.0",
        media_id="audio:123",
        payload_type="marker",
        extent_type="time",
        time=[0.1],
        description="onset",
        sandbox={"experiment_id": "exp-42"},
    )
    assert validate(ann) is True


def test_validate_dict():
    data = {
        "bopp_version": "1.0",
        "media_id": "audio:123",
        "payload": {"payload_type": "marker", "description": "onset"},
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
        "payload": {"payload_type": "marker", "description": 123},
    }
    with pytest.raises(BoppValidationError, match="Validation failed"):
        validate(invalid_data, target_type=Annotation)


def test_repr_mimebundle():
    ann = create(
        bopp_version="1.0",
        media_id="audio:repr_test",
        payload_type="marker",
        extent_type="time",
        time=[0.25, 0.75],
        description="onset",
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
    grouping = Grouping(
        name="sections", ids=[0, 1, 0, 1, 2], disjoint=False, exhaustive=False
    )
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
            disjoint=False,
            exhaustive=False,
        )
    assert grouping.labels == {"verse": 0, "chorus": 1, "bridge": 2}


def test_grouping_negative_ids_allowed_unmapped():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        grouping = Grouping(
            name="sections",
            ids=[-1, 0, -1, 1],
            labels={"verse": 0, "chorus": 1},
            disjoint=False,
            exhaustive=False,
        )
    assert grouping.ids == [-1, 0, -1, 1]


def test_grouping_warns_unmapped_non_negative_ids():
    with pytest.warns(
        UserWarning,
        match="contains non-negative group IDs not included in label mapping: \\[2\\]",
    ):
        Grouping(
            name="sections",
            ids=[0, 1, 2],
            labels={"verse": 0, "chorus": 1},
            disjoint=False,
            exhaustive=False,
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
            disjoint=False,
            exhaustive=False,
        )


def test_grouping_non_list_ids_ignored_in_root_lengths():
    class CustomGrouping(msgspec.Struct):
        name: str
        ids: Any = None

    class CustomAnnotation(BoppBase):
        media_id: str
        payload: BeatPositionPayload
        groupings: list[Any]

    ann = CustomAnnotation(
        media_id="audio:1",
        payload=BeatPositionPayload(value=[1, 2]),
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
            payload=MarkerPayload(description="test"),
            groupings=[
                Grouping(name="sections", ids=[0, 1], disjoint=False, exhaustive=False),
                Grouping(name="sections", ids=[1, 0], disjoint=False, exhaustive=False),
            ],
        )


def test_create_with_metadata_and_groupings():
    # 1. metadata_type + kwargs
    ann1 = create(
        media_id="audio:1",
        payload_type="marker",
        metadata_type="human",
        annotator_id="user_1",
        tool="sonic-visualiser",
        description="beats",
        groupings=[
            Grouping(name="beats", ids=[0, 1], disjoint=False, exhaustive=False)
        ],
    )
    assert isinstance(ann1.metadata, HumanAnnotationMetadata)
    assert ann1.metadata.annotator_id == "user_1"
    assert ann1.metadata.tool == "sonic-visualiser"
    assert len(ann1.groupings) == 1
    assert ann1.groupings[0].name == "beats"

    # 2. metadata as dict + groupings as list of dicts
    ann2 = create(
        media_id="audio:1",
        payload_type="marker",
        metadata={
            "metadata_type": "human",
            "annotator_id": "user_2",
            "tool": "audacity",
        },
        groupings=[{"name": "sections", "ids": [0, 1], "labels": {"a": 0, "b": 1}}],
        description="sections",
    )
    assert isinstance(ann2.metadata, HumanAnnotationMetadata)
    assert ann2.metadata.annotator_id == "user_2"
    assert isinstance(ann2.groupings[0], Grouping)
    assert ann2.groupings[0].name == "sections"

    # 3. metadata as struct instance
    ann3 = create(
        media_id="audio:1",
        payload_type="marker",
        metadata=HumanAnnotationMetadata(annotator_id="user_3", tool="manual"),
        description="manual",
    )
    assert ann3.metadata.annotator_id == "user_3"


def test_create_metadata_errors():
    with pytest.raises(
        BoppArgumentError, match="Cannot specify both 'metadata' and 'metadata_type'"
    ):
        create(
            media_id="audio:1",
            payload_type="marker",
            metadata_type="human",
            metadata=HumanAnnotationMetadata(annotator_id="user_1", tool="manual"),
            description="test",
        )

    with pytest.raises(
        BoppRegistryError, match="Unrecognized metadata type: 'nonexistent'"
    ):
        create(
            media_id="audio:1",
            payload_type="marker",
            metadata_type="nonexistent",
            description="test",
        )

    with pytest.raises(
        BoppRegistryError, match="Unrecognized metadata type: 'bad_type'"
    ):
        create(
            media_id="audio:1",
            payload_type="marker",
            metadata={"metadata_type": "bad_type"},
            description="test",
        )


def test_create_marker_payload_without_extent():
    ann = create(
        bopp_version="1.0",
        media_id="audio:global_marker",
        payload_type="marker",
        description="global_event",
    )
    assert ann.payload.description == "global_event"
    assert ann.extent is msgspec.UNSET
    assert validate(ann) is True


def test_create_chord_unicode_accidentals():
    ann = create(
        media_id="audio:chord_unicode",
        payload_type="chord",
        extent_type="time_interval",
        time=[0.0, 1.0, 2.0, 3.0, 4.0],
        duration=[1.0, 1.0, 1.0, 1.0, 1.0],
        value=["C♯:maj", "D♭:min", "F𝄪:maj7", "B𝄫:dim", "A:min(♮7)"],
    )
    assert validate(ann) is True

    # Invalid chord should raise validation error
    bad_chord = create(
        media_id="audio:chord_invalid",
        payload_type="chord",
        extent_type="time_interval",
        time=[0.0],
        duration=[1.0],
        value=["H:invalid"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_chord)


def test_create_key_mode_unicode_accidentals():
    ann = create(
        media_id="audio:key_unicode",
        payload_type="key_mode",
        extent_type="time_interval",
        time=[0.0, 1.0, 2.0, 3.0],
        duration=[1.0, 1.0, 1.0, 1.0],
        value=["C♯:minor", "D♭:dorian", "F𝄪:lydian", "B𝄫:mixolydian"],
    )
    assert validate(ann) is True

    # Invalid key mode
    bad_key = create(
        media_id="audio:key_invalid",
        payload_type="key_mode",
        extent_type="time_interval",
        time=[0.0],
        duration=[1.0],
        value=["InvalidKey"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_key)


def test_create_note_payload():
    ann = create(
        media_id="audio:note_spn",
        payload_type="note",
        extent_type="time",
        time=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        value=["C4", "A4", "C♯4", "E♭3", "F𝄪5", "B𝄫2", "D♮4", "C-1"],
    )
    assert validate(ann) is True

    # Invalid SPN notes
    bad_note1 = create(
        media_id="audio:note_invalid",
        payload_type="note",
        extent_type="time",
        time=[0.0],
        value=["H4"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_note1)

    bad_note2 = create(
        media_id="audio:note_invalid",
        payload_type="note",
        extent_type="time",
        time=[0.0],
        value=["C"],  # missing octave
    )
    with pytest.raises(BoppValidationError):
        validate(bad_note2)


def test_create_lyrics_payload():
    # Valid BCP 47 language tags
    for lang in ("en", "en-US", "es-419", "zh-Hans", "und"):
        ann = create(
            media_id="audio:lyrics",
            payload_type="lyrics",
            extent_type="time",
            time=[0.1],
            value=["hello"],
            language=lang,
        )
        assert ann.payload.language == lang
        assert validate(ann) is True

    # Missing mandatory language
    with pytest.raises((BoppValidationError, TypeError)):
        create(
            media_id="audio:lyrics",
            payload_type="lyrics",
            extent_type="time",
            time=[0.1],
            value=["hello"],
        )

    # Invalid BCP 47 tags
    for invalid_lang in ("123", "en_US", "e"):
        bad_lyrics = create(
            media_id="audio:lyrics",
            payload_type="lyrics",
            extent_type="time",
            time=[0.1],
            value=["hello"],
            language=invalid_lang,
        )
        with pytest.raises(BoppValidationError):
            validate(bad_lyrics)


def test_create_frequency_hz_payload():
    # Frequency with value alone
    ann = create(
        media_id="audio:freq",
        payload_type="frequency_hz",
        extent_type="time",
        time=[0.1, 0.2],
        value=[440.0, 880.0],
    )
    assert ann.payload.voiced is msgspec.UNSET
    assert validate(ann) is True

    # Frequency with voiced boolean vector
    ann_voiced = create(
        media_id="audio:freq_voiced",
        payload_type="frequency_hz",
        extent_type="time",
        time=[0.1, 0.2],
        value=[440.0, 0.0],
        voiced=[True, False],
    )
    assert ann_voiced.payload.voiced == [True, False]
    assert validate(ann_voiced) is True

    # Mismatched voiced length raises BoppArrayError
    with pytest.raises(BoppArrayError):
        create(
            media_id="audio:freq_mismatch",
            payload_type="frequency_hz",
            extent_type="time",
            time=[0.1, 0.2],
            value=[440.0, 880.0],
            voiced=[True],
        )


def test_create_object_payload_with_optional_description():
    # Without description
    ann = create(
        media_id="audio:obj",
        payload_type="object",
        extent_type="time",
        time=[0.1],
        value=[{"feature": 1}],
    )
    assert ann.payload.description is msgspec.UNSET
    assert validate(ann) is True

    # With description
    ann_desc = create(
        media_id="audio:obj_desc",
        payload_type="object",
        extent_type="time",
        time=[0.1],
        value=[{"feature": 1}],
        description="spectral features",
    )
    assert ann_desc.payload.description == "spectral features"
    assert validate(ann_desc) is True


def test_create_tag_payload():
    ann = create(
        media_id="audio:tag",
        payload_type="tag",
        extent_type="time",
        time=[0.1, 0.5],
        value=["rock", "guitar"],
    )
    assert ann.payload.value == ["rock", "guitar"]
    assert validate(ann) is True


def test_create_agreement_confidence():
    ann = create(
        media_id="audio:agreement",
        payload_type="tag",
        extent_type="time",
        time=[0.1, 0.5],
        value=["vocal", "speech"],
        confidence_type="agreement",
        confidence=[0.8, 1.0],
        num_annotators=[5, 5],
        num_annotators_common=5,
    )
    assert ann.confidence is not None
    assert ann.confidence.__struct_config__.tag == "agreement"
    assert ann.confidence.confidence == [0.8, 1.0]
    assert ann.confidence.num_annotators == [5, 5]
    assert ann.confidence.num_annotators_common == 5
    assert validate(ann) is True

    # Array length mismatch between confidence and num_annotators
    with pytest.raises(BoppArrayError):
        create(
            media_id="audio:agreement_err",
            payload_type="tag",
            extent_type="time",
            time=[0.1, 0.5],
            value=["vocal", "speech"],
            confidence_type="agreement",
            confidence=[0.8, 1.0],
            num_annotators=[5],
        )


def test_create_relation_payload():
    ann = create(
        media_id="audio:relation",
        payload_type="relation",
        performance_id="perf:100",
        value=["work:1", "work:2"],
        relation=["cover", "quote"],
        extent_type="time",
        time=[0.0, 10.0],
    )
    assert ann.payload.__struct_config__.tag == "relation"
    assert ann.payload.performance_id == "perf:100"
    assert ann.payload.value == ["work:1", "work:2"]
    assert ann.payload.relation == ["cover", "quote"]
    assert validate(ann) is True

    # Array length mismatch between value and relation
    with pytest.raises(BoppArrayError):
        create(
            media_id="audio:relation_err",
            payload_type="relation",
            performance_id="perf:100",
            value=["work:1", "work:2"],
            relation=["cover"],
            extent_type="time",
            time=[0.0, 10.0],
        )


def test_create_with_types():
    ann = create(
        media_id="audio:types",
        payload_type="tag",
        value=["rock"],
        extent_type="time",
        time=[1.0],
        confidence_type="likelihood",
        confidence=[0.95],
        metadata_type="human",
        annotator_id="h1",
        tool="manual",
    )
    assert ann.payload.__struct_config__.tag == "tag"
    assert ann.extent.__struct_config__.tag == "time"
    assert ann.confidence.__struct_config__.tag == "likelihood"
    assert ann.metadata.__struct_config__.tag == "human"
    assert validate(ann) is True


def test_create_missing_payload_type():
    with pytest.raises(TypeError, match="payload_type"):
        create(media_id="audio:missing")  # type: ignore[call-arg]
