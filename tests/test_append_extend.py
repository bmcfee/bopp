from __future__ import annotations

import msgspec
import pandas as pd
import polars as pl
import pytest

import bopp


def test_append_inplace_incremental_authoring():
    meta = bopp.create_metadata("human", annotator_id="alice", tool="audacity")
    ann = bopp.create(
        media_id="track:test_authoring",
        payload_type="tag",
        extent_type="time_interval",
        empty=True,
        metadata=meta,
    )
    initial_id = ann.id
    assert ann.payload.value == []
    assert ann.extent.time == []
    assert ann.extent.duration == []

    # 1. Append first observation
    res = ann.append(time=0.0, duration=2.0, value="intro", inplace=True)
    assert res is ann
    assert ann.payload.value == ["intro"]
    assert ann.extent.time == [0.0]
    assert ann.extent.duration == [2.0]
    # Voiding id field
    assert ann.id is msgspec.UNSET
    # Preserves original metadata
    assert ann.metadata == meta
    assert ann.parents is msgspec.UNSET

    # 2. Append second observation via dict
    ann.append({"time": 2.0, "duration": 3.0, "value": "verse"}, inplace=True)
    assert ann.payload.value == ["intro", "verse"]
    assert ann.extent.time == [0.0, 2.0]
    assert ann.extent.duration == [2.0, 3.0]

    # Re-seal ID and validate
    new_id = bopp.ensure_annotation_id(ann)
    assert new_id != initial_id
    assert bopp.validate(ann) is True


def test_append_out_of_place_derivation():
    ann = bopp.create(
        media_id="track:oop",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[2.0],
        value=["intro"],
    )
    parent_id = ann.id

    # Append out of place
    derived = ann.append(time=2.0, duration=2.0, value="chorus", inplace=False)

    # Original is untouched
    assert ann.id == parent_id
    assert ann.payload.value == ["intro"]

    # Derived has new observations, parent lineage, derived metadata, and new deterministic ID
    assert derived is not ann
    assert derived.payload.value == ["intro", "chorus"]
    assert derived.extent.time == [0.0, 2.0]
    assert derived.parents == [parent_id]
    assert derived.metadata.__struct_config__.tag == "derived"
    assert derived.metadata.transform == "append"
    assert derived.metadata.parameters == {"observation_index": 1}
    assert derived.id != parent_id
    assert bopp.validate_annotation_id(derived) is True
    assert bopp.validate(derived) is True


def test_append_without_derivation_metadata():
    ann = bopp.create(
        media_id="track:noop_meta",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[2.0],
        value=["intro"],
        metadata=bopp.create_metadata("human", annotator_id="alice", tool="sv"),
    )
    derived = bopp.append(
        ann, time=2.0, duration=2.0, value="verse", inplace=False, derive=False
    )
    assert derived.metadata.__struct_config__.tag == "human"
    assert derived.metadata.annotator_id == "alice"
    assert derived.parents == [ann.id]


def test_append_missing_and_unconsumed_fields():
    ann = bopp.create(
        media_id="track:err",
        payload_type="tag",
        extent_type="time_interval",
        empty=True,
    )

    # Missing duration
    with pytest.raises(
        bopp.BoppArgumentError, match="Missing required field.*extent:duration"
    ):
        ann.append(time=0.0, value="verse", inplace=True)

    # Missing value
    with pytest.raises(
        bopp.BoppArgumentError, match="Missing required field.*payload:value"
    ):
        ann.append(time=0.0, duration=1.0, inplace=True)

    # Unconsumed field
    with pytest.raises(
        bopp.BoppArgumentError, match="Unconsumed argument.*unknown_foo"
    ):
        ann.append(time=0.0, duration=1.0, value="verse", unknown_foo=42, inplace=True)


def test_append_with_groupings():
    ann = bopp.create(
        media_id="track:groups",
        payload_type="tag",
        extent_type="time_interval",
        empty=True,
        groupings=[bopp.create_grouping("layer", labels={"intro": 0, "verse": 1})],
    )

    # 1. Append with known string label as kwarg
    ann.append(time=0.0, duration=1.0, value="A", layer="intro", inplace=True)
    assert ann.groupings[0].ids == [0]

    # 2. Append with explicit groupings dict mapping
    ann.append(
        time=1.0,
        duration=1.0,
        value="B",
        groupings={"layer": "verse"},
        inplace=True,
    )
    assert ann.groupings[0].ids == [0, 1]

    # 3. Append with new string label (auto-registers into labels)
    ann.append(time=2.0, duration=1.0, value="C", layer="chorus", inplace=True)
    assert ann.groupings[0].ids == [0, 1, 2]
    assert ann.groupings[0].labels["chorus"] == 2

    # 4. Append with integer ID
    ann.append(time=3.0, duration=1.0, value="D", layer=1, inplace=True)
    assert ann.groupings[0].ids == [0, 1, 2, 1]

    # 5. Missing grouping assignment raises error
    with pytest.raises(
        bopp.BoppArgumentError, match="Missing required field.*grouping:layer"
    ):
        ann.append(time=4.0, duration=1.0, value="E", inplace=True)


def test_append_time_uniform_extent():
    ann = bopp.create(
        media_id="track:uniform",
        payload_type="tempo",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=0.5,
        value=[],
    )
    assert ann.extent.start_time == 0.0

    # time_uniform requires no extent kwargs
    ann.append(value=120.0, inplace=True)
    ann.append(value=122.0, inplace=True)
    assert ann.payload.value == [120.0, 122.0]
    bopp.ensure_annotation_id(ann)
    assert bopp.validate(ann) is True


def test_append_bounds_sanity_check_and_update():
    ann = bopp.create(
        media_id="track:bounds",
        payload_type="tag",
        extent_type="time_interval",
        min_time=0.0,
        max_time=10.0,
        time=[0.0],
        duration=[2.0],
        value=["a"],
    )

    # Exceeds max_time -> emits warning
    with pytest.warns(UserWarning, match="exceeds explicit max_time"):
        ann.append(time=9.0, duration=3.0, value="b", inplace=True)
    assert ann.extent.max_time == 10.0

    # Below min_time -> emits warning
    with pytest.warns(UserWarning, match="is below explicit min_time"):
        ann.append(time=-1.0, duration=1.0, value="c", inplace=True)

    # update_bounds=True expands explicit bounds
    ann.append(
        time=10.0,
        duration=5.0,
        value="d",
        inplace=True,
        update_bounds=True,
    )
    assert ann.extent.max_time == 15.0

    # Point extent (time) bounds
    ann_pt = bopp.create(
        media_id="track:pt_bounds",
        payload_type="tag",
        extent_type="time",
        min_time=1.0,
        max_time=10.0,
        time=[2.0],
        value=["a"],
    )
    with pytest.warns(UserWarning, match="is below explicit min_time"):
        ann_pt.append(time=0.5, value="b", inplace=True)
    with pytest.warns(UserWarning, match="exceeds explicit max_time"):
        ann_pt.append(time=11.0, value="c", inplace=True)
    ann_pt.append(time=5.0, value="d", inplace=True)

    # Min_max extent (time_frequency_box) bounds
    ann_box = bopp.create(
        media_id="track:box_bounds",
        payload_type="tag",
        extent_type="time_frequency_box",
        min_freq=100.0,
        max_freq=1000.0,
        time=[1.0],
        duration=[1.0],
        freq_min=[200.0],
        freq_max=[800.0],
        value=["a"],
    )
    with pytest.warns(UserWarning, match="is below explicit min_freq"):
        ann_box.append(
            time=2.0,
            duration=1.0,
            freq_min=50.0,
            freq_max=500.0,
            value="b",
            inplace=True,
        )
    with pytest.warns(UserWarning, match="exceeds explicit max_freq"):
        ann_box.append(
            time=3.0,
            duration=1.0,
            freq_min=500.0,
            freq_max=1200.0,
            value="c",
            inplace=True,
        )
    ann_box.append(
        time=4.0,
        duration=1.0,
        freq_min=200.0,
        freq_max=800.0,
        value="d",
        inplace=True,
    )

    # Score interval fraction bounds and update_bounds=True
    ann_frac = bopp.create(
        media_id="track:frac_bounds",
        payload_type="tag",
        extent_type="score_interval",
        min_quarter=[1, 1],
        max_quarter=[2, 1],
        quarter=[[1, 1]],
        duration=[[1, 1]],
        value=["a"],
    )
    ann_frac.append(
        quarter=[0, 1],
        duration=[1, 1],
        value="b",
        inplace=True,
        update_bounds=True,
    )
    assert ann_frac.extent.min_quarter == [0, 1]
    ann_frac.append(
        quarter=[3, 1],
        duration=[2, 1],
        value="c",
        inplace=True,
        update_bounds=True,
    )
    assert ann_frac.extent.max_quarter == [5, 1]


def test_append_disjoint_and_exhaustive_warnings():
    # disjoint=True warning on overlap with preceding observation
    ann_disj = bopp.create(
        media_id="track:disj",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[2.0],
        value=["a"],
        disjoint=True,
    )
    with pytest.warns(UserWarning, match="violating disjoint=True"):
        ann_disj.append(time=1.0, duration=2.0, value="b", inplace=True)
    # Non-violating time_interval disjoint (branch 822->857)
    ann_disj.append(time=5.0, duration=1.0, value="ok", inplace=True)

    # time extent disjoint warning and non-violating
    ann_time = bopp.create(
        media_id="track:time_disj",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["a"],
        disjoint=True,
    )
    with pytest.warns(UserWarning, match="violating disjoint=True"):
        ann_time.append(time=1.0, value="b", inplace=True)
    ann_time.append(time=3.0, value="c", inplace=True)

    # score_interval disjoint warning and non-violating
    ann_si = bopp.create(
        media_id="track:si_disj",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1]],
        duration=[[2, 1]],
        value=["a"],
        disjoint=True,
    )
    with pytest.warns(UserWarning, match="violating disjoint=True"):
        ann_si.append(quarter=[1, 1], duration=[1, 1], value="b", inplace=True)
    ann_si.append(quarter=[3, 1], duration=[1, 1], value="c", inplace=True)

    # score_quarter disjoint warning and non-violating
    ann_sq = bopp.create(
        media_id="track:sq_disj",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[2, 1]],
        value=["a"],
        disjoint=True,
    )
    with pytest.warns(UserWarning, match="violating disjoint=True"):
        ann_sq.append(quarter=[1, 1], value="b", inplace=True)
    ann_sq.append(quarter=[3, 1], value="c", inplace=True)

    # exhaustive=True warning on gap
    ann_exh = bopp.create(
        media_id="track:exh",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[2.0],
        value=["a"],
        exhaustive=True,
    )
    with pytest.warns(
        UserWarning,
        match="Gap detected before appended observation.*violating exhaustive=True",
    ):
        ann_exh.append(time=3.0, duration=2.0, value="b", inplace=True)
    # Non-violating time_interval exhaustive (branch 860->879)
    ann_exh.append(time=5.0, duration=1.0, value="ok", inplace=True)

    # score_interval exhaustive warning and non-violating
    ann_si_exh = bopp.create(
        media_id="track:si_exh",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1]],
        duration=[[2, 1]],
        value=["a"],
        exhaustive=True,
    )
    with pytest.warns(UserWarning, match="violating exhaustive=True"):
        ann_si_exh.append(quarter=[3, 1], duration=[1, 1], value="b", inplace=True)
    ann_si_exh.append(quarter=[2, 1], duration=[1, 1], value="c", inplace=True)

    # Disjoint and exhaustive flags on an extent without interval ordering (branches 847->858, 867->880)
    ann_px = bopp.create(
        media_id="track:px_disj_exh",
        payload_type="tag",
        extent_type="pixel",
        x=[0.0],
        y=[0.0],
        value=["a"],
        disjoint=True,
        exhaustive=True,
    )
    ann_px.append(x=1.0, y=1.0, value="b", inplace=True)


def test_append_grouping_disjoint_warning():
    ann = bopp.create(
        media_id="track:grp_disj",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 5.0],
        duration=[2.0, 2.0],
        value=["a", "b"],
        groupings=[bopp.create_grouping("layer", [0, 1], disjoint=True)],
    )
    # Group 0 last ended at 2.0. Appending overlapping observation for group 0 at 1.0 warns
    with pytest.warns(UserWarning, match="violating disjoint=True"):
        ann.append(time=1.0, duration=2.0, value="c", layer=0, inplace=True)

    # Appending group 2 which has never appeared before (branch 885->879)
    ann.append(time=1.0, duration=2.0, value="d", layer=2, inplace=True)

    # Appending group 0 at 10.0 which has no overlap (branch 888->894)
    ann.append(time=10.0, duration=2.0, value="e", layer=0, inplace=True)


def test_extend_with_another_annotation_inplace_and_out_of_place():
    ann1 = bopp.create(
        media_id="track:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=["intro", "verse"],
        groupings=[bopp.create_grouping("sec", ["A", "B"])],
    )
    ann2 = bopp.create(
        media_id="track:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[4.0],
        duration=[2.0],
        value=["chorus"],
        groupings=[bopp.create_grouping("sec", ["C"])],
    )

    # Out-of-place extend
    derived = bopp.extend(ann1, ann2, inplace=False)
    assert derived is not ann1
    assert derived.payload.value == ["intro", "verse", "chorus"]
    assert derived.extent.time == [0.0, 2.0, 4.0]
    assert derived.groupings[0].ids == [0, 1, 0]
    assert derived.parents == [ann1.id, ann2.id]
    assert derived.metadata.__struct_config__.tag == "derived"
    assert derived.metadata.transform == "extend"

    # In-place extend
    res = ann1.extend(ann2, inplace=True)
    assert res is ann1
    assert ann1.payload.value == ["intro", "verse", "chorus"]
    assert ann1.id is msgspec.UNSET


def test_extend_with_sequence_of_dicts():
    ann = bopp.create(
        media_id="track:seq",
        payload_type="tag",
        extent_type="time_interval",
        empty=True,
    )
    records = [
        {"time": 0.0, "duration": 1.0, "value": "a"},
        {"time": 1.0, "duration": 1.0, "value": "b"},
        {"time": 2.0, "duration": 1.5, "value": "c"},
    ]
    derived = ann.extend(records, inplace=False)
    assert derived.payload.value == ["a", "b", "c"]
    assert derived.extent.time == [0.0, 1.0, 2.0]
    assert derived.extent.duration == [1.0, 1.0, 1.5]
    assert derived.metadata.transform == "extend"
    assert derived.metadata.parameters == {"count": 3}


def test_extend_with_columnar_dict():
    ann = bopp.create(
        media_id="track:col",
        payload_type="tag",
        extent_type="time_interval",
        empty=True,
    )
    col_dict = {
        "time": [0.0, 1.0],
        "duration": [1.0, 2.0],
        "value": ["v1", "v2"],
    }
    ann.extend(col_dict, inplace=True)
    assert ann.payload.value == ["v1", "v2"]
    assert ann.extent.time == [0.0, 1.0]

    # Mismatched array length raises BoppArrayError
    with pytest.raises(bopp.BoppArrayError, match="Mismatched array lengths"):
        ann.extend({"time": [2.0], "duration": [1.0, 2.0], "value": ["v3"]})


def test_extend_with_dataframes():
    # Pandas DataFrame
    ann_pandas = bopp.create(
        media_id="track:df",
        payload_type="tag",
        extent_type="time_interval",
        empty=True,
    )
    df_pd = pd.DataFrame(
        {
            "time": [0.0, 2.0],
            "duration": [2.0, 2.0],
            "value": ["p1", "p2"],
        }
    )
    ann_pandas.extend(df_pd, inplace=True)
    assert ann_pandas.payload.value == ["p1", "p2"]

    # Polars DataFrame
    ann_polars = bopp.create(
        media_id="track:df",
        payload_type="tag",
        extent_type="time_interval",
        empty=True,
    )
    df_pl = pl.DataFrame(
        {
            "time": [10.0, 12.0],
            "duration": [2.0, 3.0],
            "value": ["pl1", "pl2"],
        }
    )
    derived = ann_polars.extend(df_pl, inplace=False)
    assert derived.payload.value == ["pl1", "pl2"]
    assert derived.extent.time == [10.0, 12.0]


def test_extend_mismatch_errors():
    ann1 = bopp.create(
        media_id="track:1",
        payload_type="tag",
        value=["a"],
    )
    ann_diff_media = bopp.create(
        media_id="track:2",
        payload_type="tag",
        value=["b"],
    )
    with pytest.raises(bopp.BoppArgumentError, match="media_id mismatch"):
        ann1.extend(ann_diff_media)

    ann_diff_payload = bopp.create(
        media_id="track:1",
        payload_type="beat",
        value=[1.0],
    )
    with pytest.raises(bopp.BoppArgumentError, match="Payload type mismatch"):
        ann1.extend(ann_diff_payload)


def test_append_groupings_arguments_and_error_handling():
    ann = bopp.create(
        media_id="track:grp_args",
        payload_type="tag",
        extent_type="time",
        time=[0.0],
        value=["a"],
        groupings=[bopp.create_grouping("sec", [0])],
    )
    # Valid sequence of groupings
    ann.append(time=1.0, value="b", groupings=[0], inplace=True)
    assert ann.groupings[0].ids == [0, 0]

    # Invalid sequence length
    with pytest.raises(
        bopp.BoppArgumentError, match="Expected 1 grouping values, got 2"
    ):
        ann.append(time=2.0, value="c", groupings=[0, 1])

    # Invalid groupings type
    with pytest.raises(
        bopp.BoppArgumentError, match="groupings must be a dictionary or sequence"
    ):
        ann.append(time=2.0, value="c", groupings=123)

    # String grouping value when labels is UNSET
    ann.append(time=3.0, value="d", groupings={"sec": "label1"}, inplace=True)
    assert ann.groupings[0].labels == {"label1": 0}

    # Non-integer grouping ID
    with pytest.raises(bopp.BoppArgumentError, match="must be an integer"):
        ann.append(time=4.0, value="e", groupings={"sec": object()})


def test_append_with_confidence_facet():
    ann = bopp.create(
        media_id="track:conf_append",
        payload_type="tag",
        extent_type="time_interval",
        confidence_type="likelihood",
        time=[0.0],
        duration=[1.0],
        value=["a"],
        confidence=[0.9],
        groupings=[bopp.create_grouping("sec", [0])],
    )
    # Missing confidence field
    with pytest.raises(bopp.BoppArgumentError, match="confidence:confidence"):
        ann.append(time=1.0, duration=1.0, value="b", sec=0)

    # inplace=True with confidence (line 902)
    ann.append(time=1.0, duration=1.0, value="b", confidence=0.8, sec=0, inplace=True)
    assert ann.confidence.confidence == [0.9, 0.8]

    # inplace=False with confidence & groupings (lines 921, 923)
    derived = ann.append(
        time=2.0, duration=1.0, value="c", confidence=0.7, sec=0, inplace=False
    )
    assert derived.confidence.confidence == [0.9, 0.8, 0.7]
    assert derived.groupings[0].ids == [0, 0, 0]


def test_extend_edge_cases_and_coverage():
    a1 = bopp.create(
        media_id="m", payload_type="tag", extent_type="time", time=[0.0], value=["a"]
    )
    # Extent mismatch
    a_ext = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.0],
        duration=[1.0],
        value=["b"],
    )
    with pytest.raises(bopp.BoppArgumentError, match="Extent type mismatch"):
        a1.extend(a_ext)

    # bopp_version mismatch
    a_ver = bopp.create(
        media_id="m", payload_type="tag", extent_type="time", time=[1.0], value=["b"]
    )
    a_ver.bopp_version = "v99"
    with pytest.raises(bopp.BoppArgumentError, match="bopp_version mismatch"):
        a1.extend(a_ver)

    # Confidence mismatch
    a_conf1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        confidence=[0.9],
        time=[0.0],
        value=["a"],
    )
    a_conf2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        confidence_type="variance",
        confidence=[0.1],
        time=[1.0],
        value=["b"],
    )
    with pytest.raises(bopp.BoppArgumentError, match="Confidence type mismatch"):
        a_conf1.extend(a_conf2)

    # Both have confidence (line 1047)
    a_conf3 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        confidence_type="likelihood",
        confidence=[0.8],
        time=[1.0],
        value=["b"],
    )
    ext_conf = a_conf1.extend(a_conf3, inplace=False)
    assert ext_conf.confidence.confidence == [0.9, 0.8]

    # Grouping layer count mismatch
    a_grp1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[0.0],
        value=["a"],
        groupings=[bopp.create_grouping("g1", [0])],
    )
    a_grp2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["b"],
        groupings=[
            bopp.create_grouping("g1", [0]),
            bopp.create_grouping("g2", [0]),
        ],
    )
    with pytest.raises(bopp.BoppArgumentError, match="Grouping layer count mismatch"):
        a_grp1.extend(a_grp2)

    # Grouping name mismatch
    a_grp3 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["b"],
        groupings=[bopp.create_grouping("diff_name", [0])],
    )
    with pytest.raises(bopp.BoppArgumentError, match="Grouping name mismatch"):
        a_grp1.extend(a_grp3)

    # Target grouping has UNSET labels, src has labels (line 1056)
    a_unlbl = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[0.0],
        value=["a"],
        groupings=[bopp.create_grouping("g", [0])],
    )
    a_lbl = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["b"],
        groupings=[bopp.create_grouping("g", [1], labels={"x": 1})],
    )
    ext_lbl = a_unlbl.extend(a_lbl, inplace=False)
    assert ext_lbl.groupings[0].labels == {"x": 1}

    # Conflicting label mapping (line 1060)
    a_lbl_conflict = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["b"],
        groupings=[bopp.create_grouping("g", [0], labels={"x": 2})],
    )
    with pytest.raises(bopp.BoppArgumentError, match="Conflicting label mapping"):
        a_lbl.extend(a_lbl_conflict)

    # Source grouping has UNSET labels (branch 1054->1064)
    ext_lbl2 = a_lbl.extend(a_unlbl, inplace=False)
    assert ext_lbl2.groupings[0].labels == {"x": 1}

    # Second annotation has UNSET id (branch 1071->1073)
    a2_no_id = bopp.create(
        media_id="m", payload_type="tag", extent_type="time", time=[1.0], value=["b"]
    )
    a2_no_id.id = msgspec.UNSET
    ext_no_id = a1.extend(a2_no_id, inplace=False)
    assert ext_no_id.parents == [a1.id]

    # derive=False with another annotation (branch 1075->1083)
    meta = bopp.create_metadata("human", annotator_id="alice", tool="sv")
    a1_meta = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[0.0],
        value=["a"],
        metadata=meta,
    )
    ext_noderive = a1_meta.extend(a1, inplace=False, derive=False)
    assert ext_noderive.metadata == meta

    # derive=False with list of dicts (branch 1124->1131)
    ext_seq_noderive = a1_meta.extend(
        [{"time": 2.0, "value": "c"}], inplace=False, derive=False
    )
    assert ext_seq_noderive.metadata == meta

    # Unsupported observations type (line 1135)
    with pytest.raises(bopp.BoppArgumentError, match="Unsupported observations type"):
        a1.extend(42)
