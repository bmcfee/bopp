from __future__ import annotations

import msgspec
import pytest

import bopp


def test_create_metadata_all_types():
    # 1. human
    meta_human = bopp.create_metadata(
        "human", annotator_id="alice", tool="sonic-visualiser"
    )
    assert meta_human.__struct_config__.tag == "human"
    assert meta_human.annotator_id == "alice"
    assert meta_human.tool == "sonic-visualiser"

    # 2. algorithm
    meta_algo = bopp.create_metadata(
        "algorithm",
        algorithm_id="beat-track",
        version="1.0.0",
        parameters={"hop_length": 512},
    )
    assert meta_algo.__struct_config__.tag == "algorithm"
    assert meta_algo.algorithm_id == "beat-track"
    assert meta_algo.version == "1.0.0"
    assert meta_algo.parameters == {"hop_length": 512}

    # 3. derived
    meta_derived = bopp.create_metadata(
        "derived",
        transform="trim",
        parameters={"start": 0.0, "end": 10.0},
        software="bopp==0.0.1",
        description="trimmed to 10s",
    )
    assert meta_derived.__struct_config__.tag == "derived"
    assert meta_derived.transform == "trim"
    assert meta_derived.software == "bopp==0.0.1"

    # 4. crowd
    meta_crowd = bopp.create_metadata(
        "crowd",
        platform="mturk",
        num_annotators=5,
        consensus="majority_vote",
    )
    assert meta_crowd.__struct_config__.tag == "crowd"
    assert meta_crowd.num_annotators == 5

    # 5. sensor
    meta_sensor = bopp.create_metadata(
        "sensor",
        device="mic-array-1",
        settings={"gain": 0.8},
    )
    assert meta_sensor.__struct_config__.tag == "sensor"
    assert meta_sensor.device == "mic-array-1"

    # 6. other
    meta_other = bopp.create_metadata("other", notes="annotated in studio")
    assert meta_other.__struct_config__.tag == "other"
    assert meta_other.notes == "annotated in studio"


def test_create_metadata_from_dict():
    meta = bopp.create_metadata(
        {"metadata_type": "human", "annotator_id": "bob", "tool": "audacity"}
    )
    assert meta.__struct_config__.tag == "human"
    assert meta.annotator_id == "bob"

    # Dictionary with kwargs override
    meta2 = bopp.create_metadata({"metadata_type": "other"}, notes="passed as kwarg")
    assert meta2.notes == "passed as kwarg"


def test_create_metadata_errors():
    with pytest.raises(bopp.BoppArgumentError, match="must contain 'metadata_type'"):
        bopp.create_metadata({"annotator_id": "alice"})

    with pytest.raises(bopp.BoppRegistryError, match="Unrecognized metadata type"):
        bopp.create_metadata("nonexistent_type")

    with pytest.raises(bopp.BoppArgumentError, match="Failed to instantiate"):
        bopp.create_metadata("human", invalid_field="foo")

    with pytest.raises(
        bopp.BoppValidationError, match="Validation failed for metadata 'human'"
    ):
        bopp.create_metadata({"metadata_type": "human", "annotator_id": [1, 2]})


def test_create_grouping_integer_ids():
    g = bopp.create_grouping(
        "sections",
        [0, 1, 0, 2],
        labels={"verse": 0, "chorus": 1, "bridge": 2},
        disjoint=True,
        exhaustive=False,
    )
    assert g.name == "sections"
    assert g.ids == [0, 1, 0, 2]
    assert g.labels == {"verse": 0, "chorus": 1, "bridge": 2}
    assert g.disjoint is True
    assert g.exhaustive is False


def test_create_grouping_string_labels_auto_categorize():
    # Pass string labels as first argument (ids)
    g = bopp.create_grouping("voices", ["soprano", "alto", "soprano", "tenor"])
    assert g.name == "voices"
    assert g.ids == [0, 1, 0, 2]
    assert g.labels == {"soprano": 0, "alto": 1, "tenor": 2}

    # Pass string labels via labels keyword argument
    g2 = bopp.create_grouping("structure", labels=["intro", "verse", "chorus", "verse"])
    assert g2.ids == [0, 1, 2, 1]
    assert g2.labels == {"intro": 0, "verse": 1, "chorus": 2}


def test_create_grouping_string_labels_with_explicit_mapping():
    g = bopp.create_grouping(
        "sections",
        ["chorus", "verse"],
        labels={"verse": 10, "chorus": 20},
    )
    assert g.ids == [20, 10]
    assert g.labels == {"verse": 10, "chorus": 20}

    with pytest.raises(
        bopp.BoppArgumentError, match="not found in the provided labels"
    ):
        bopp.create_grouping(
            "sections",
            ["bridge"],
            labels={"verse": 0, "chorus": 1},
        )


def test_create_grouping_empty_and_unlabeled():
    g_empty = bopp.create_grouping("empty_layer")
    assert g_empty.ids == []
    assert g_empty.labels is msgspec.UNSET

    g_unlabeled = bopp.create_grouping("layer", [1, 2, 3])
    assert g_unlabeled.ids == [1, 2, 3]
    assert g_unlabeled.labels is msgspec.UNSET


def test_create_grouping_errors():
    with pytest.raises(bopp.BoppArgumentError, match="must be integers or strings"):
        bopp.create_grouping("invalid", [object()])

    with pytest.raises(bopp.BoppArgumentError, match="must be integers or strings"):
        bopp.create_grouping("invalid_float", [1.9])

    with pytest.raises(bopp.BoppArgumentError, match="must be integers or strings"):
        bopp.create_grouping("invalid_bool", [True])

    with pytest.raises(bopp.BoppArgumentError, match="must be integers or strings"):
        bopp.create_grouping("invalid_mixed", [0, "1"])

    with pytest.raises(
        bopp.BoppValidationError, match="Failed to create Grouping 'sec'"
    ):
        bopp.create_grouping("sec", [0], labels={"a": [1, 2]})


def test_create_sandbox():
    s1 = bopp.create_sandbox(run_id=42, status="ok")
    assert s1 == {"run_id": 42, "status": "ok"}

    s2 = bopp.create_sandbox({"flag": True}, extra="data")
    assert s2 == {"flag": True, "extra": "data"}

    with pytest.raises(bopp.BoppArgumentError, match="must be a dictionary"):
        bopp.create_sandbox("not a dict")  # type: ignore[arg-type]

    with pytest.raises(bopp.BoppValidationError, match="cannot be serialized to JSON"):
        bopp.create_sandbox(bad_value=object())


def test_create_with_empty_flag():
    ann = bopp.create(
        media_id="audio:empty_test",
        payload_type="tag",
        extent_type="time_interval",
        empty=True,
    )
    assert ann.payload.value == []
    assert ann.extent.time == []
    assert ann.extent.duration == []
    assert bopp.validate(ann) is True


def test_create_with_single_grouping_and_metadata_dict():
    # Pass singular grouping as dict and metadata_type with dict parameters
    ann = bopp.create(
        media_id="audio:test_single",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0],
        value=["a", "b"],
        metadata_type="human",
        metadata={"annotator_id": "alice", "tool": "sonic-visualiser"},
        grouping={"name": "sections", "ids": ["intro", "verse"]},
    )
    assert ann.metadata.annotator_id == "alice"
    assert len(ann.groupings) == 1
    assert ann.groupings[0].name == "sections"
    assert ann.groupings[0].ids == [0, 1]
    assert ann.groupings[0].labels == {"intro": 0, "verse": 1}
    assert bopp.validate(ann) is True


def test_create_grouping_and_metadata_helpers_integration():
    meta = bopp.create_metadata(
        "algorithm", algorithm_id="alg-1", version="1.0", parameters={}
    )
    grp = bopp.create_grouping("phrases", ["p1", "p2"])
    box = bopp.create_sandbox(test_key="value")

    ann = bopp.create(
        media_id="audio:full_helpers",
        payload_type="note",
        extent_type="time",
        time=[0.0, 1.0],
        value=["C4", "D4"],
        metadata=meta,
        groupings=grp,
        sandbox=box,
    )
    assert ann.metadata == meta
    assert ann.groupings[0] == grp
    assert ann.sandbox == box
    assert bopp.validate(ann) is True


def test_registry_getters():
    assert bopp.get_registry() is not None
    assert bopp.get_payload_class("tag") is not None
    assert bopp.get_extent_class("time") is not None
    assert bopp.get_confidence_class("likelihood") is not None
    assert bopp.get_metadata_class("human") is not None
    assert bopp.get_grouping_class() is not None
    assert bopp.get_annotation_class() is not None


def test_create_edge_cases_and_errors():
    # 1. Both grouping and groupings
    g1 = bopp.create_grouping("g1", [0])
    g2 = bopp.create_grouping("g2", [0])
    with pytest.raises(
        bopp.BoppArgumentError, match="Cannot specify both 'grouping' and 'groupings'"
    ):
        bopp.create(
            media_id="m",
            payload_type="tag",
            value=["a"],
            grouping=g1,
            groupings=[g2],
        )

    # 2. empty=True with explicit payload field provided (hits 471->470 branch)
    ann_empty_val = bopp.create(media_id="m", payload_type="tag", empty=True, value=[])
    assert ann_empty_val.payload.value == []

    # 3. empty=True with confidence_type (hits lines 501-506)
    ann_conf = bopp.create(
        media_id="m", payload_type="tag", confidence_type="likelihood", empty=True
    )
    assert ann_conf.confidence.confidence == []

    # 3b. empty=True with confidence list already provided (branch 502->501)
    ann_conf2 = bopp.create(
        media_id="m",
        payload_type="tag",
        confidence_type="likelihood",
        confidence=[],
        empty=True,
    )
    assert ann_conf2.confidence.confidence == []

    # 4. Conflicting metadata_type vs dict metadata_type (line 521)
    with pytest.raises(bopp.BoppArgumentError, match="Conflicting metadata_type"):
        bopp.create(
            media_id="m",
            payload_type="tag",
            empty=True,
            metadata_type="human",
            metadata={"metadata_type": "algorithm"},
        )

    # 5. Invalid kwargs in metadata dict (lines 528-529)
    with pytest.raises(
        bopp.BoppArgumentError, match="Failed to instantiate metadata 'human'"
    ):
        bopp.create(
            media_id="m",
            payload_type="tag",
            empty=True,
            metadata_type="human",
            metadata={"invalid_key": 123},
        )

    # 6. resolve_ext=True with payload_type="ext" (line 580)
    with pytest.warns(UserWarning, match="No extension registered"):
        ann_ext = bopp.create(
            media_id="m",
            payload_type="ext",
            ext_schema="http://example.com/schema",
            value=[{"name": "item1", "score": 1.0}],
            resolve_ext=True,
        )
    assert ann_ext.payload.value == [{"name": "item1", "score": 1.0}]


def test_validate_variants():
    # Target type validation with non-struct, non-dict object
    assert bopp.validate("hello", target_type=str) is True
    assert bopp.validate(42, target_type=int) is True

    # Raw dict containing msgspec.UNSET filtered out
    tag_cls = bopp.get_payload_class("tag")
    assert (
        bopp.validate(
            {"value": ["a"], "confidence": msgspec.UNSET}, target_type=tag_cls
        )
        is True
    )
