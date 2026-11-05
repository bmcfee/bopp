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

    # 6. Out-of-place append does not mutate original grouping labels
    ann_orig = bopp.create(
        media_id="track:orig_grp",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[1.0],
        value=["A"],
        groupings=[bopp.create_grouping("layer", [0], labels={"intro": 0})],
    )
    derived = ann_orig.append(
        time=1.0, duration=1.0, value="B", layer="bridge", inplace=False
    )
    assert "bridge" not in ann_orig.groupings[0].labels
    assert ann_orig.groupings[0].labels == {"intro": 0}
    assert derived.groupings[0].labels == {"intro": 0, "bridge": 1}
    assert derived.groupings[0].ids == [0, 1]


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

    # Out-of-place append with update_bounds=True does not mutate original annotation
    ann_orig_bound = bopp.create(
        media_id="track:orig_bound",
        payload_type="tag",
        extent_type="time",
        time=[0.0],
        value=["a"],
        max_time=1.0,
    )
    orig_id = ann_orig_bound.id
    der_bound = ann_orig_bound.append(
        time=5.0, value="b", inplace=False, update_bounds=True
    )
    assert ann_orig_bound.extent.max_time == 1.0
    assert ann_orig_bound.id == orig_id
    assert der_bound.extent.max_time == 5.0
    assert der_bound.id != orig_id

    # Time uniform extent bounds checking and updating in append
    ann_u = bopp.create(
        media_id="track:u_bounds",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        max_time=1.0,
        value=["a", "b"],
    )
    with pytest.warns(UserWarning, match="exceeds explicit max_time"):
        ann_u.append(value="c", inplace=False)
    der_u = ann_u.append(value="c", inplace=False, update_bounds=True)
    assert ann_u.extent.max_time == 1.0
    assert der_u.extent.max_time == 2.0

    ann_empty_u = bopp.create(
        media_id="track:empty_u",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        min_time=1.0,
        empty=True,
    )
    with pytest.warns(UserWarning, match="is below explicit min_time"):
        ann_empty_u.append(value="a", inplace=False)
    der_min_u = ann_empty_u.append(value="a", inplace=False, update_bounds=True)
    assert der_min_u.extent.min_time == 0.0

    # Global annotation (no extent) append and extend
    ann_global = bopp.create(media_id="track:g", payload_type="tag", value=["a"])
    ann_global.append(value="b", inplace=True)
    assert ann_global.payload.value == ["a", "b"]
    ann_global_2 = bopp.create(media_id="track:g", payload_type="tag", value=["c"])
    ext_g = ann_global.extend(ann_global_2, inplace=False)
    assert ext_g.payload.value == ["a", "b", "c"]
    assert ext_g.extent is msgspec.UNSET

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
    assert derived.groupings[0].ids == [0, 1, 2]
    assert derived.groupings[0].labels == {"A": 0, "B": 1, "C": 2}
    assert derived.parents == [ann1.id, ann2.id]
    assert derived.metadata.__struct_config__.tag == "derived"
    assert derived.metadata.transform == "extend"

    # In-place extend
    res = ann1.extend(ann2, inplace=True)
    assert res is ann1
    assert ann1.payload.value == ["intro", "verse", "chorus"]
    assert ann1.groupings[0].ids == [0, 1, 2]
    assert ann1.groupings[0].labels == {"A": 0, "B": 1, "C": 2}
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
    with pytest.raises(bopp.BoppArgumentError, match="expected integer or string"):
        ann.append(time=4.0, value="e", groupings={"sec": object()})

    with pytest.raises(bopp.BoppArgumentError, match="expected integer or string"):
        ann.append(time=4.0, value="e", groupings={"sec": 1.9})

    with pytest.raises(bopp.BoppArgumentError, match="expected integer or string"):
        ann.append(time=4.0, value="e", groupings={"sec": True})

    # Unknown grouping layer in dict
    with pytest.raises(bopp.BoppArgumentError, match="Unknown grouping layer"):
        ann.append(time=4.0, value="e", groupings={"sec": 0, "typo_layer": 1})


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


def test_extend_edge_cases_and_coverage(monkeypatch):
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

    # Remapping disjoint and conflicting IDs when merging labels
    g_base = bopp.create_grouping("g", [0], labels={"A": 0})
    a_base = bopp.create(
        media_id="track:base",
        payload_type="tag",
        extent_type="time",
        time=[0.0],
        value=["a"],
        groupings=[g_base],
    )
    g_other = bopp.create_grouping("g", [0, 1], labels={"B": 0, "A": 1})
    a_other = bopp.create(
        media_id="track:base",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0],
        value=["b", "c"],
        groupings=[g_other],
    )
    ext_remap = a_base.extend(a_other, inplace=False)
    assert ext_remap.groupings[0].labels == {"A": 0, "B": 1}
    assert ext_remap.groupings[0].ids == [0, 1, 0]
    assert bopp.validate(ext_remap) is True

    # Same IDs in target and source for the same label: no remap needed
    g_same = bopp.create_grouping("g", [0], labels={"A": 0})
    a_same = bopp.create(
        media_id="track:base",
        payload_type="tag",
        extent_type="time",
        time=[3.0],
        value=["d"],
        groupings=[g_same],
    )
    ext_same = a_base.extend(a_same, inplace=False)
    assert ext_same.groupings[0].labels == {"A": 0}
    assert ext_same.groupings[0].ids == [0, 0]

    # Source label ID not in used_ids: adopted directly
    g_unused = bopp.create_grouping("g", [5], labels={"C": 5})
    a_unused = bopp.create(
        media_id="track:base",
        payload_type="tag",
        extent_type="time",
        time=[4.0],
        value=["e"],
        groupings=[g_unused],
    )
    ext_unused = a_base.extend(a_unused, inplace=False)
    assert ext_unused.groupings[0].labels == {"A": 0, "C": 5}
    assert ext_unused.groupings[0].ids == [0, 5]

    # Inplace extend with another annotation
    ext_inplace = a_base.extend(a_same, inplace=True)
    assert ext_inplace is a_base
    assert a_base.id is msgspec.UNSET

    # Annotation-to-annotation bounds checking and update_bounds
    a_b1 = bopp.create(
        media_id="track:base",
        payload_type="tag",
        extent_type="time",
        time=[0.0],
        value=["a"],
        min_time=1.0,
        max_time=2.0,
    )
    a_b2 = bopp.create(
        media_id="track:base",
        payload_type="tag",
        extent_type="time",
        time=[0.5, 3.0],
        value=["b", "c"],
    )
    with pytest.warns(UserWarning, match="is below explicit min_time"):
        a_b1.extend(a_b2, inplace=False)
    with pytest.warns(UserWarning, match="exceeds explicit max_time"):
        a_b1.extend(a_b2, inplace=False)

    der_ext_b = a_b1.extend(a_b2, inplace=False, update_bounds=True)
    assert a_b1.extent.min_time == 1.0
    assert a_b1.extent.max_time == 2.0
    assert der_ext_b.extent.min_time == 0.5
    assert der_ext_b.extent.max_time == 3.0

    # Fraction extent bounds in extend
    a_frac1 = bopp.create(
        media_id="track:base",
        payload_type="tag",
        extent_type="score_interval",
        min_quarter=[1, 1],
        max_quarter=[3, 1],
        quarter=[[1, 1]],
        duration=[[1, 1]],
        value=["a"],
    )
    a_frac2 = bopp.create(
        media_id="track:base",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1], [3, 1]],
        duration=[[1, 1], [2, 1]],
        value=["b", "c"],
    )
    der_frac_ext = a_frac1.extend(a_frac2, inplace=False, update_bounds=True)
    assert der_frac_ext.extent.min_quarter == [0, 1]
    assert der_frac_ext.extent.max_quarter == [5, 1]

    # Time uniform extent in extend
    ann_u1 = bopp.create(
        media_id="track:base",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        max_time=1.0,
        value=["a"],
    )
    ann_u2 = bopp.create(
        media_id="track:base",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=1.0,
        sampling_period=1.0,
        value=["b", "c"],
    )
    with pytest.warns(UserWarning, match="exceeds explicit max_time"):
        ann_u1.extend(ann_u2, inplace=False)
    der_u_ext = ann_u1.extend(ann_u2, inplace=False, update_bounds=True)
    assert der_u_ext.extent.max_time == 2.0

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

    # Extend within explicit bounds (branches 1097->1112 and 1115->1089)
    ann_b1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.0],
        duration=[1.0],
        value=["a"],
        min_time=0.0,
        max_time=10.0,
    )
    ann_b2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[2.0],
        duration=[1.0],
        value=["b"],
    )
    ext_within = ann_b1.extend(ann_b2, inplace=False)
    assert ext_within.extent.min_time == 0.0
    assert ext_within.extent.max_time == 10.0

    # Non-columnar settings mismatch tests
    # Scalar payload field mismatch
    ann_p_scalar1 = bopp.create(
        media_id="m",
        payload_type="lyrics",
        extent_type="time",
        time=[0.0],
        value=["hello"],
        language="en",
    )
    ann_p_scalar2 = bopp.create(
        media_id="m",
        payload_type="lyrics",
        extent_type="time",
        time=[1.0],
        value=["bonjour"],
        language="fr",
    )
    with pytest.raises(
        bopp.BoppArgumentError, match="Scalar payload field 'language' mismatch"
    ):
        ann_p_scalar1.extend(ann_p_scalar2)

    # Scalar confidence field mismatch
    ann_c_scalar1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        time=[0.0],
        value=["a"],
        confidence=[1.0],
        num_annotators_common=5,
    )
    ann_c_scalar2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        time=[1.0],
        value=["b"],
        confidence=[0.8],
        num_annotators_common=3,
    )
    with pytest.raises(
        bopp.BoppArgumentError,
        match="Scalar confidence field 'num_annotators_common' mismatch",
    ):
        ann_c_scalar1.extend(ann_c_scalar2)

    # Uniform extent sampling_period and start_time mismatch
    ann_u_sp1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        value=["a"],
    )
    ann_u_sp2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=1.0,
        sampling_period=2.0,
        value=["b"],
    )
    with pytest.raises(bopp.BoppArgumentError, match="sampling_period mismatch"):
        ann_u_sp1.extend(ann_u_sp2)

    ann_u_st_bad = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=5.0,
        sampling_period=1.0,
        value=["b"],
    )
    with pytest.raises(bopp.BoppArgumentError, match="start_time mismatch"):
        ann_u_sp1.extend(ann_u_st_bad)

    # Extent disjoint & exhaustive settings mismatch
    ann_disj1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[1.0],
        value=["a"],
        disjoint=True,
    )
    ann_disj2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.0],
        duration=[1.0],
        value=["b"],
        disjoint=False,
    )
    with pytest.raises(
        bopp.BoppArgumentError, match="Extent disjoint setting mismatch"
    ):
        ann_disj1.extend(ann_disj2)

    ann_exh1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[1.0],
        value=["a"],
        exhaustive=True,
    )
    ann_exh2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.0],
        duration=[1.0],
        value=["b"],
        exhaustive=False,
    )
    with pytest.raises(
        bopp.BoppArgumentError, match="Extent exhaustive setting mismatch"
    ):
        ann_exh1.extend(ann_exh2)

    # Grouping disjoint & exhaustive settings mismatch
    ann_grp_d1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[0.0],
        value=["a"],
        groupings=[bopp.create_grouping("layer", [0], disjoint=True)],
    )
    ann_grp_d2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["b"],
        groupings=[bopp.create_grouping("layer", [0], disjoint=False)],
    )
    with pytest.raises(
        bopp.BoppArgumentError, match="Grouping 'layer' disjoint setting mismatch"
    ):
        ann_grp_d1.extend(ann_grp_d2)

    ann_grp_e1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[0.0],
        value=["a"],
        groupings=[bopp.create_grouping("layer", [0], exhaustive=True)],
    )
    ann_grp_e2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["b"],
        groupings=[bopp.create_grouping("layer", [0], exhaustive=False)],
    )
    with pytest.raises(
        bopp.BoppArgumentError, match="Grouping 'layer' exhaustive setting mismatch"
    ):
        ann_grp_e1.extend(ann_grp_e2)

    # Optional column mismatch when n_obs_target > 0
    ann_opt1 = bopp.create(
        media_id="m",
        payload_type="relation",
        extent_type="time",
        time=[0.0],
        performance_id="perf1",
        value=["a"],
    )
    ann_opt2 = bopp.create(
        media_id="m",
        payload_type="relation",
        extent_type="time",
        time=[1.0],
        performance_id="perf1",
        value=["b"],
        relation=["r1"],
    )
    with pytest.raises(bopp.BoppArgumentError, match="Payload column mismatch"):
        ann_opt1.extend(ann_opt2)

    # Empty target adopting optional columns and non-columnar settings
    ann_empty_rel = bopp.create(
        media_id="m",
        payload_type="relation",
        extent_type="time_interval",
        performance_id="perf1",
        empty=True,
    )
    ann_full_rel = bopp.create(
        media_id="m",
        payload_type="relation",
        extent_type="time_interval",
        time=[0.0],
        duration=[1.0],
        performance_id="perf1",
        value=["a"],
        relation=["r1"],
        disjoint=True,
        exhaustive=True,
    )
    ext_from_empty = ann_empty_rel.extend(ann_full_rel, inplace=False)
    assert ext_from_empty.payload.relation == ["r1"]
    assert ext_from_empty.extent.disjoint is True
    assert ext_from_empty.extent.exhaustive is True

    # Sanity checks warning: disjoint violation when extending from another annotation
    ann_dj_a = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[2.0],
        value=["a"],
        disjoint=True,
    )
    ann_dj_b = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.0],
        duration=[2.0],
        value=["b"],
        disjoint=True,
    )
    with pytest.warns(UserWarning, match="violating disjoint=True"):
        ann_dj_a.extend(ann_dj_b)

    # Sanity checks warning: point time disjoint violation
    ann_pt_a = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["a"],
        disjoint=True,
    )
    ann_pt_b = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["b"],
        disjoint=True,
    )
    with pytest.warns(UserWarning, match="violating disjoint=True"):
        ann_pt_a.extend(ann_pt_b)

    # Sanity checks warning: score_interval disjoint violation
    ann_si_a = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1]],
        duration=[[2, 1]],
        value=["a"],
        disjoint=True,
    )
    ann_si_b = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[1, 1]],
        duration=[[1, 1]],
        value=["b"],
        disjoint=True,
    )
    with pytest.warns(UserWarning, match="violating disjoint=True"):
        ann_si_a.extend(ann_si_b)

    # Sanity checks warning: score_quarter disjoint violation
    ann_sq_a = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[2, 1]],
        value=["a"],
        disjoint=True,
    )
    ann_sq_b = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[1, 1]],
        value=["b"],
        disjoint=True,
    )
    with pytest.warns(UserWarning, match="violating disjoint=True"):
        ann_sq_a.extend(ann_sq_b)

    # Sanity checks warning: exhaustive gap violation
    ann_ex_a = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[1.0],
        value=["a"],
        exhaustive=True,
    )
    ann_ex_b = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[2.0],
        duration=[1.0],
        value=["b"],
        exhaustive=True,
    )
    with pytest.warns(UserWarning, match="violating exhaustive=True"):
        ann_ex_a.extend(ann_ex_b)

    # Sanity checks warning: score_interval exhaustive gap violation
    ann_siex_a = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1]],
        duration=[[1, 1]],
        value=["a"],
        exhaustive=True,
    )
    ann_siex_b = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[3, 1]],
        duration=[[1, 1]],
        value=["b"],
        exhaustive=True,
    )
    with pytest.warns(UserWarning, match="violating exhaustive=True"):
        ann_siex_a.extend(ann_siex_b)

    # Sanity checks warning: grouping layer disjoint violation
    ann_gdj_a = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[2.0],
        value=["a"],
        groupings=[bopp.create_grouping("layer", [0], disjoint=True)],
    )
    ann_gdj_b = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.0],
        duration=[2.0],
        value=["b"],
        groupings=[bopp.create_grouping("layer", [0], disjoint=True)],
    )
    with pytest.warns(UserWarning, match="overlaps with previous group member"):
        ann_gdj_a.extend(ann_gdj_b)

    # Clean non-violating extensions for disjoint, exhaustive, and grouping disjoint
    ann_clean1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0],
        duration=[2.0],
        value=["a"],
        disjoint=True,
        exhaustive=True,
        groupings=[bopp.create_grouping("layer", [0], disjoint=True)],
    )
    ann_clean2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_interval",
        time=[2.0],
        duration=[2.0],
        value=["b"],
        disjoint=True,
        exhaustive=True,
        groupings=[bopp.create_grouping("layer", [0], disjoint=True)],
    )
    ext_clean = ann_clean1.extend(ann_clean2, inplace=False)
    assert ext_clean.extent.time == [0.0, 2.0]

    # Clean score_interval extension
    ann_si_c1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1]],
        duration=[[2, 1]],
        value=["a"],
        disjoint=True,
        exhaustive=True,
    )
    ann_si_c2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[2, 1]],
        duration=[[2, 1]],
        value=["b"],
        disjoint=True,
        exhaustive=True,
    )
    ext_si_clean = ann_si_c1.extend(ann_si_c2, inplace=False)
    assert len(ext_si_clean.extent.quarter) == 2

    # Clean score_quarter and point time extensions
    ann_sq_c1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[1, 1]],
        value=["a"],
        disjoint=True,
    )
    ann_sq_c2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_quarter",
        quarter=[[2, 1]],
        value=["b"],
        disjoint=True,
    )
    ann_sq_c1.extend(ann_sq_c2, inplace=False)

    ann_pt_c1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["a"],
        disjoint=True,
    )
    ann_pt_c2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[2.0],
        value=["b"],
        disjoint=True,
    )
    ann_pt_c1.extend(ann_pt_c2, inplace=False)

    # Empty uniform extent adopting settings
    ann_empty_u = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=0.5,
        empty=True,
        groupings=[bopp.create_grouping("layer", ids=[], disjoint=True)],
    )
    ann_src_u = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=0.5,
        value=["a", "b"],
        groupings=[bopp.create_grouping("layer", [0, 0], disjoint=True)],
    )
    ext_u = ann_empty_u.extend(ann_src_u, inplace=False)
    assert ext_u.extent.start_time == 0.0
    assert ext_u.extent.sampling_period == 0.5

    # Scalar confidence match and optional confidence reconciliation on empty
    ann_empty_c = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        empty=True,
    )
    ann_full_c = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        time=[0.0],
        value=["a"],
        confidence=[0.9],
        num_annotators_common=10,
        num_annotators=[5],
    )
    ext_c = ann_empty_c.extend(ann_full_c, inplace=False)
    assert ext_c.confidence.num_annotators == [5]
    assert ext_c.confidence.num_annotators_common == 10

    # Confidence optional column mismatch when n_obs_target > 0
    ann_c_mismatch1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        time=[0.0],
        value=["a"],
        confidence=[0.9],
        num_annotators_common=10,
    )
    with pytest.raises(bopp.BoppArgumentError, match="Confidence column mismatch"):
        ann_c_mismatch1.extend(ann_full_c)

    # Empty target adopting grouping disjoint/exhaustive
    ann_empty_lyr = bopp.create(
        media_id="m",
        payload_type="lyrics",
        extent_type="time",
        language="en",
        empty=True,
        groupings=[
            bopp.create_grouping("layer", ids=[], disjoint=False, exhaustive=False)
        ],
    )
    ann_full_lyr = bopp.create(
        media_id="m",
        payload_type="lyrics",
        extent_type="time",
        time=[0.0],
        value=["hello"],
        language="en",
        groupings=[bopp.create_grouping("layer", [0], disjoint=True, exhaustive=True)],
    )
    ext_lyr = ann_empty_lyr.extend(ann_full_lyr, inplace=False)
    assert ext_lyr.payload.language == "en"
    assert ext_lyr.groupings[0].disjoint is True
    assert ext_lyr.groupings[0].exhaustive is True

    # Empty target with conflicting explicit scalar payload
    ann_empty_lyr_conflict = bopp.create(
        media_id="m",
        payload_type="lyrics",
        extent_type="time",
        language="fr",
        empty=True,
        groupings=[bopp.create_grouping("layer", ids=[])],
    )
    with pytest.raises(
        bopp.BoppArgumentError, match="Scalar payload field 'language' mismatch"
    ):
        ann_empty_lyr_conflict.extend(ann_full_lyr)

    # Empty target adopting optional scalar payload field (e.g. object.description)
    ann_empty_obj = bopp.create(
        media_id="m",
        payload_type="object",
        extent_type="time",
        empty=True,
    )
    assert ann_empty_obj.payload.description is msgspec.UNSET
    ann_full_obj = bopp.create(
        media_id="m",
        payload_type="object",
        extent_type="time",
        time=[0.0],
        value=[{"track": 1}],
        description="my object description",
    )
    ext_obj = ann_empty_obj.extend(ann_full_obj, inplace=False)
    assert ext_obj.payload.description == "my object description"

    # Empty target reconciling optional payload column
    ann_empty_rel = bopp.create(
        media_id="m",
        payload_type="relation",
        extent_type="time",
        performance_id="p1",
        empty=True,
    )
    ann_empty_rel.payload.relation = msgspec.UNSET
    ann_full_rel = bopp.create(
        media_id="m",
        payload_type="relation",
        extent_type="time",
        performance_id="p1",
        time=[0.0],
        value=["w1"],
        relation=["cover"],
    )
    ext_rel = ann_empty_rel.extend(ann_full_rel, inplace=False)
    assert ext_rel.payload.relation == ["cover"]

    # Empty target reconciling optional confidence column
    ann_empty_agree = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        empty=True,
    )
    ann_empty_agree.confidence.num_annotators = msgspec.UNSET
    ann_full_agree = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        confidence_type="agreement",
        time=[0.0],
        value=["a"],
        confidence=[0.9],
        num_annotators=[5],
    )
    ext_agree = ann_empty_agree.extend(ann_full_agree, inplace=False)
    assert ext_agree.confidence.num_annotators == [5]

    # Empty target extending another empty annotation
    ext_both_empty = ann_empty_rel.extend(ann_empty_rel, inplace=False)
    assert len(ext_both_empty.extent.time) == 0

    # Exhaustive check on score_interval with consecutive observations
    ann_score1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1]],
        duration=[[1, 1]],
        value=["a"],
        exhaustive=True,
    )
    ann_score2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[1, 1]],
        duration=[[1, 1]],
        value=["b"],
        exhaustive=True,
    )
    ext_score = ann_score1.extend(ann_score2, inplace=False)
    assert len(ext_score.extent.quarter) == 2

    # Exhaustive check on time extent (tag != time_interval and tag != score_interval)
    ann_t_ex1 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[0.0],
        value=["a"],
        exhaustive=True,
    )
    ann_t_ex2 = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["b"],
        exhaustive=True,
    )
    ext_t_ex = ann_t_ex1.extend(ann_t_ex2, inplace=False)
    assert len(ext_t_ex.extent.time) == 2

    # Extent column mismatch test (line 1127)
    orig_get_cols = bopp.core._get_annotation_columns

    def fake_get_cols(ann):
        p, e, c, g = orig_get_cols(ann)
        if ann is a_ext_col_src:
            e = dict(e)
            e["extra_col"] = []
        return p, e, c, g

    a_ext_col_src = bopp.create(
        media_id="m",
        payload_type="tag",
        extent_type="time",
        time=[1.0],
        value=["b"],
    )
    monkeypatch.setattr(bopp.core, "_get_annotation_columns", fake_get_cols)
    with pytest.raises(bopp.BoppArgumentError, match="Extent column mismatch"):
        a1.extend(a_ext_col_src)
    monkeypatch.undo()

    # Annotation with no payload (branch 1042->1052)
    fake_ann1 = bopp.create(media_id="m", payload_type="tag", empty=True)
    fake_ann2 = bopp.create(media_id="m", payload_type="tag", empty=True)
    fake_ann1.payload = None
    fake_ann2.payload = None
    ext_no_payload = bopp.extend(fake_ann1, fake_ann2, inplace=False)
    assert ext_no_payload.payload is None

    # Unsupported observations type (line 1135)
    with pytest.raises(bopp.BoppArgumentError, match="Unsupported observations type"):
        a1.extend(42)
