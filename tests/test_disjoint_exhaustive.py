import pytest

from bopp.core import create, validate
from bopp.exceptions import BoppValidationError
from bopp.io import (
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.util import (
    get_extent_bounds,
    infer_extent_bounds,
    is_disjoint,
    is_exhaustive,
)

# =====================================================================
# 1. Disjointness Tests
# =====================================================================


def test_disjoint_time_points():
    # Unique time points
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time",
        time=[0.1, 0.5, 1.2],
        value=["a", "b", "c"],
        disjoint=True,
    )
    assert is_disjoint(ann) is True
    assert validate(ann) is True

    # Duplicate time points with disjoint=True raises BoppValidationError
    with pytest.raises(
        BoppValidationError, match="marked disjoint=True, but observations overlap"
    ):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time",
            time=[0.1, 0.5, 0.5],
            value=["a", "b", "c"],
            disjoint=True,
        )

    # Duplicate time points with disjoint=False is allowed
    ann_non_disjoint = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time",
        time=[0.1, 0.5, 0.5],
        value=["a", "b", "c"],
        disjoint=False,
    )
    assert is_disjoint(ann_non_disjoint) is False
    assert validate(ann_non_disjoint) is True


def test_disjoint_time_intervals():
    # Abutting non-overlapping intervals [0, 1), [1, 3), [3, 5)
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 1.0, 3.0],
        duration=[1.0, 2.0, 2.0],
        value=["a", "b", "c"],
        disjoint=True,
    )
    assert is_disjoint(ann) is True
    assert validate(ann) is True

    # Unsorted abutting intervals [3, 5), [0, 1), [1, 3)
    ann_unsorted = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[3.0, 0.0, 1.0],
        duration=[2.0, 1.0, 2.0],
        value=["c", "a", "b"],
        disjoint=True,
    )
    assert is_disjoint(ann_unsorted) is True
    assert validate(ann_unsorted) is True

    # Overlapping intervals [0, 2) and [1, 3)
    with pytest.raises(
        BoppValidationError, match="marked disjoint=True, but observations overlap"
    ):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_interval",
            time=[0.0, 1.0],
            duration=[2.0, 2.0],
            value=["a", "b"],
            disjoint=True,
        )

    # Contained interval [0, 5) and [1, 2)
    with pytest.raises(
        BoppValidationError, match="marked disjoint=True, but observations overlap"
    ):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_interval",
            time=[0.0, 1.0],
            duration=[5.0, 1.0],
            value=["a", "b"],
            disjoint=True,
        )


def test_disjoint_time_uniform():
    # Uniform time extent always has disjoint=True
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=0.1,
        value=["a", "b", "c"],
    )
    assert ann.extent.disjoint is True
    assert is_disjoint(ann) is True
    assert validate(ann) is True

    # Setting disjoint=False on time_uniform is disallowed
    with pytest.raises(
        BoppValidationError, match="Uniform time extent requires disjoint=True"
    ):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_uniform",
            start_time=0.0,
            sampling_period=0.1,
            disjoint=False,
            value=["a", "b", "c"],
        )


def test_disjoint_score_fraction_intervals():
    # Score intervals using exact rational fractions [0, 1/2) and [1/2, 1)
    ann = create(
        media_id="score:1",
        payload_type="tag",
        extent_type="score_interval",
        quarter=[[0, 1], [1, 2]],
        duration=[[1, 2], [1, 2]],
        value=["note1", "note2"],
        disjoint=True,
    )
    assert is_disjoint(ann) is True
    assert validate(ann) is True

    # Overlapping fractions [0, 3/4) and [1/2, 1)
    with pytest.raises(
        BoppValidationError, match="marked disjoint=True, but observations overlap"
    ):
        create(
            media_id="score:1",
            payload_type="tag",
            extent_type="score_interval",
            quarter=[[0, 1], [1, 2]],
            duration=[[3, 4], [1, 2]],
            value=["note1", "note2"],
            disjoint=True,
        )


def test_disjoint_pixel_and_boxes():
    # Pixel 2D points: unique coordinates
    ann_pix = create(
        media_id="img:1",
        payload_type="tag",
        extent_type="pixel",
        x=[10.0, 20.0, 10.0],
        y=[10.0, 20.0, 30.0],
        value=["p1", "p2", "p3"],
        disjoint=True,
    )
    assert is_disjoint(ann_pix) is True
    assert validate(ann_pix) is True

    # Duplicate pixel points
    with pytest.raises(
        BoppValidationError, match="marked disjoint=True, but observations overlap"
    ):
        create(
            media_id="img:1",
            payload_type="tag",
            extent_type="pixel",
            x=[10.0, 20.0, 10.0],
            y=[10.0, 20.0, 10.0],
            value=["p1", "p2", "p3"],
            disjoint=True,
        )

    # Pixel boxes: non-overlapping
    ann_box = create(
        media_id="img:1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0, 10.0],
        width=[10.0, 10.0],
        y=[0.0, 0.0],
        height=[10.0, 10.0],
        value=["b1", "b2"],
        disjoint=True,
    )
    assert is_disjoint(ann_box) is True
    assert validate(ann_box) is True

    # Pixel boxes overlapping in both dimensions
    with pytest.raises(
        BoppValidationError, match="marked disjoint=True, but observations overlap"
    ):
        create(
            media_id="img:1",
            payload_type="tag",
            extent_type="pixel_box",
            x=[0.0, 5.0],
            width=[10.0, 10.0],
            y=[0.0, 5.0],
            height=[10.0, 10.0],
            value=["b1", "b2"],
            disjoint=True,
        )


def test_disjoint_spatiotemporal_intervals():
    # Cartesian time interval: intervals overlap in time, but at DIFFERENT spatial locations
    ann = create(
        media_id="spatial:1",
        payload_type="tag",
        extent_type="cartesian_time_interval",
        time=[0.0, 0.5],
        duration=[2.0, 2.0],
        x=[0.0, 10.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        value=["obj1", "obj2"],
        disjoint=True,
    )
    # Different spatial positions -> disjoint in spacetime!
    assert is_disjoint(ann) is True
    assert validate(ann) is True

    # Intervals overlap in time at the EXACT SAME spatial location -> collision!
    with pytest.raises(
        BoppValidationError, match="marked disjoint=True, but observations overlap"
    ):
        create(
            media_id="spatial:1",
            payload_type="tag",
            extent_type="cartesian_time_interval",
            time=[0.0, 0.5],
            duration=[2.0, 2.0],
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            value=["obj1", "obj2"],
            disjoint=True,
        )


# =====================================================================
# 2. Exhaustiveness Tests
# =====================================================================


def test_exhaustive_time_intervals_explicit_bounds():
    # Intervals [0, 4), [4, 7), [7, 10) fully cover [0.0, 10.0]
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 4.0, 7.0],
        duration=[4.0, 3.0, 3.0],
        value=["a", "b", "c"],
        exhaustive=True,
        min_time=0.0,
        max_time=10.0,
    )
    assert is_exhaustive(ann) is True
    assert validate(ann) is True

    # Gap at start: [1, 4), [4, 10) does not cover [0, 10)
    with pytest.raises(BoppValidationError, match="marked exhaustive=True"):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_interval",
            time=[1.0, 4.0],
            duration=[3.0, 6.0],
            value=["a", "b"],
            exhaustive=True,
            min_time=0.0,
            max_time=10.0,
        )

    # Gap in middle: [0, 4), [5, 10) does not cover [0, 10)
    with pytest.raises(BoppValidationError, match="marked exhaustive=True"):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_interval",
            time=[0.0, 5.0],
            duration=[4.0, 5.0],
            value=["a", "b"],
            exhaustive=True,
            min_time=0.0,
            max_time=10.0,
        )

    # Gap at end: [0, 4), [4, 8) does not cover [0, 10)
    with pytest.raises(BoppValidationError, match="marked exhaustive=True"):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_interval",
            time=[0.0, 4.0],
            duration=[4.0, 4.0],
            value=["a", "b"],
            exhaustive=True,
            min_time=0.0,
            max_time=10.0,
        )


def test_exhaustive_time_intervals_inferred_bounds():
    # Contiguous intervals [2.0, 5.0) and [5.0, 9.0) with omitted bounds
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[2.0, 5.0],
        duration=[3.0, 4.0],
        value=["a", "b"],
        exhaustive=True,
    )
    bounds = get_extent_bounds(ann)
    assert bounds["time"] == (2.0, 9.0)
    assert is_exhaustive(ann) is True
    assert validate(ann) is True

    # Partial explicit bounds: min_time=0.0 specified, max_time inferred
    with pytest.raises(BoppValidationError, match="marked exhaustive=True"):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_interval",
            time=[2.0, 5.0],
            duration=[3.0, 4.0],
            value=["a", "b"],
            min_time=0.0,
            exhaustive=True,
        )


def test_exhaustive_overlapping_intervals():
    # Intervals overlap [0, 6) and [4, 10), but completely cover [0, 10]
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 4.0],
        duration=[6.0, 6.0],
        value=["a", "b"],
        disjoint=False,
        exhaustive=True,
        min_time=0.0,
        max_time=10.0,
    )
    assert is_disjoint(ann) is False
    assert is_exhaustive(ann) is True
    assert validate(ann) is True


def test_exhaustive_empty_extents():
    # Empty extent with min_time == max_time is valid
    ann_empty_equal = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[],
        duration=[],
        value=[],
        min_time=5.0,
        max_time=5.0,
        exhaustive=True,
    )
    assert is_exhaustive(ann_empty_equal) is True
    assert validate(ann_empty_equal) is True

    # Empty extent with omitted bounds is valid
    ann_empty_omitted = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[],
        duration=[],
        value=[],
        exhaustive=True,
    )
    assert is_exhaustive(ann_empty_omitted) is True
    assert validate(ann_empty_omitted) is True

    # Empty extent with min_time < max_time cannot cover the range
    with pytest.raises(BoppValidationError, match="marked exhaustive=True"):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_interval",
            time=[],
            duration=[],
            value=[],
            min_time=0.0,
            max_time=10.0,
            exhaustive=True,
        )


def test_exhaustive_continuous_points():
    # Points spanning min_time to max_time
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time",
        time=[0.0, 2.5, 5.0, 10.0],
        value=["a", "b", "c", "d"],
        min_time=0.0,
        max_time=10.0,
        exhaustive=True,
    )
    assert is_exhaustive(ann) is True
    assert validate(ann) is True

    # Points failing to span down to min_time
    with pytest.raises(BoppValidationError, match="marked exhaustive=True"):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time",
            time=[1.0, 5.0, 10.0],
            value=["a", "b", "c"],
            min_time=0.0,
            max_time=10.0,
            exhaustive=True,
        )


def test_exhaustive_discrete_text():
    # Text character offsets covering [0, 3] with all integers
    ann = create(
        media_id="text:1",
        payload_type="tag",
        extent_type="text",
        offset=[0, 1, 2, 3],
        value=["w", "o", "r", "d"],
        min_offset=0,
        max_offset=3,
        exhaustive=True,
    )
    assert is_exhaustive(ann) is True
    assert validate(ann) is True

    # Missing integer offset 2
    with pytest.raises(BoppValidationError, match="marked exhaustive=True"):
        create(
            media_id="text:1",
            payload_type="tag",
            extent_type="text",
            offset=[0, 1, 3],
            value=["w", "o", "d"],
            min_offset=0,
            max_offset=3,
            exhaustive=True,
        )


def test_exhaustive_2d_boxes():
    # Two boxes covering [0, 10] x [0, 10] without gaps
    ann = create(
        media_id="img:1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0, 5.0],
        width=[5.0, 5.0],
        y=[0.0, 0.0],
        height=[10.0, 10.0],
        value=["left", "right"],
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=10.0,
        exhaustive=True,
    )
    assert is_exhaustive(ann) is True
    assert validate(ann) is True

    # Gap in coverage [x from 5 to 6 is missing]
    with pytest.raises(BoppValidationError, match="marked exhaustive=True"):
        create(
            media_id="img:1",
            payload_type="tag",
            extent_type="pixel_box",
            x=[0.0, 6.0],
            width=[5.0, 4.0],
            y=[0.0, 0.0],
            height=[10.0, 10.0],
            value=["left", "right"],
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            exhaustive=True,
        )


def test_get_and_infer_extent_bounds():
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.5, 4.0],
        duration=[2.0, 3.5],
        value=["a", "b"],
        min_time=0.0,
    )
    # Inferred directly from data: min is 1.5, max is 4.0 + 3.5 = 7.5
    assert infer_extent_bounds(ann) == {"time": (1.5, 7.5)}
    # Effective bounds combine explicit min_time=0.0 with inferred max_time=7.5
    assert get_extent_bounds(ann) == {"time": (0.0, 7.5)}
    # Subset by indices: [0] has time=1.5, dur=2.0 -> (1.5, 3.5)
    assert infer_extent_bounds(ann, indices=[0]) == {"time": (1.5, 3.5)}
    assert get_extent_bounds(ann, indices=[0]) == {"time": (0.0, 3.5)}


# =====================================================================
# 3. Groupings Tests
# =====================================================================


def test_grouping_disjoint():
    # Overall extent has overlapping intervals:
    # Group 0: [0, 2), [2, 4)  -> disjoint
    # Group 1: [1, 3), [3, 5)  -> disjoint
    # Combined: [0, 2), [1, 3), [2, 4), [3, 5) -> NOT disjoint overall!
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 1.0, 2.0, 3.0],
        duration=[2.0, 2.0, 2.0, 2.0],
        value=["g0_a", "g1_a", "g0_b", "g1_b"],
        disjoint=False,
        groupings=[
            {
                "name": "tracks",
                "ids": [0, 1, 0, 1],
                "disjoint": True,
                "exhaustive": False,
            }
        ],
    )
    assert is_disjoint(ann) is False
    assert is_disjoint(ann, grouping=ann.groupings[0]) is True
    assert validate(ann) is True

    # Now introduce an overlap within Group 0:
    with pytest.raises(
        BoppValidationError, match="Grouping 'tracks' is marked disjoint=True"
    ):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_interval",
            time=[0.0, 1.0, 1.5, 3.0],
            duration=[2.0, 2.0, 2.0, 2.0],
            value=["g0_a", "g1_a", "g0_b", "g1_b"],
            disjoint=False,
            groupings=[
                {
                    "name": "tracks",
                    "ids": [0, 1, 0, 1],
                    "disjoint": True,
                    "exhaustive": False,
                }
            ],
        )


def test_grouping_negative_ids_ignored():
    # Negative IDs (e.g. -1 for unassigned) overlapping with each other are ignored
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 0.5, 2.0],
        duration=[2.0, 2.0, 2.0],
        value=["unassigned1", "unassigned2", "valid_group"],
        disjoint=False,
        groupings=[
            {
                "name": "tracks",
                "ids": [-1, -1, 0],
                "disjoint": True,
                "exhaustive": False,
            }
        ],
    )
    assert is_disjoint(ann, grouping=ann.groupings[0]) is True
    assert validate(ann) is True


def test_grouping_exhaustive():
    # Bounds: [0.0, 6.0]
    # Group 0: [0, 3), [3, 6) -> covers [0, 6]
    # Group 1: [0, 2), [2, 6) -> covers [0, 6]
    ann = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 0.0, 3.0, 2.0],
        duration=[3.0, 2.0, 3.0, 4.0],
        value=["g0_1", "g1_1", "g0_2", "g1_2"],
        min_time=0.0,
        max_time=6.0,
        groupings=[
            {
                "name": "layers",
                "ids": [0, 1, 0, 1],
                "disjoint": True,
                "exhaustive": True,
            }
        ],
    )
    assert is_exhaustive(ann, grouping=ann.groupings[0]) is True
    assert validate(ann) is True

    # Group 1 has a gap (only covers [0, 2) and [3, 6), missing [2, 3))
    with pytest.raises(
        BoppValidationError, match="Grouping 'layers' is marked exhaustive=True"
    ):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_interval",
            time=[0.0, 0.0, 3.0, 3.0],
            duration=[3.0, 2.0, 3.0, 3.0],
            value=["g0_1", "g1_1", "g0_2", "g1_2"],
            min_time=0.0,
            max_time=6.0,
            groupings=[
                {
                    "name": "layers",
                    "ids": [0, 1, 0, 1],
                    "disjoint": True,
                    "exhaustive": True,
                }
            ],
        )


# =====================================================================
# 4. Serialization Roundtrip Tests
# =====================================================================


def test_serialization_roundtrip_json_msgpack_csv(tmp_path):
    ann = create(
        media_id="audio:track1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 2.0],
        duration=[2.0, 3.0],
        value=["verse", "chorus"],
        disjoint=True,
        exhaustive=True,
        min_time=0.0,
        max_time=5.0,
        groupings=[
            {"name": "sections", "ids": [0, 0], "disjoint": True, "exhaustive": True}
        ],
    )

    # 1. JSON roundtrip
    json_path = tmp_path / "test.bopp"
    save_bopp_json(ann, json_path)
    loaded_json = load_bopp_json(json_path)
    assert loaded_json.extent.disjoint is True
    assert loaded_json.extent.exhaustive is True
    assert loaded_json.extent.min_time == 0.0
    assert loaded_json.extent.max_time == 5.0
    assert loaded_json.groupings[0].disjoint is True
    assert loaded_json.groupings[0].exhaustive is True
    assert validate(loaded_json) is True

    # 2. MsgPack roundtrip
    msgpack_path = tmp_path / "test.bopp.msgpack"
    save_bopp_msgpack(ann, msgpack_path)
    loaded_msgpack = load_bopp_msgpack(msgpack_path)
    assert loaded_msgpack.extent.disjoint is True
    assert loaded_msgpack.extent.exhaustive is True
    assert loaded_msgpack.extent.min_time == 0.0
    assert loaded_msgpack.extent.max_time == 5.0
    assert loaded_msgpack.groupings[0].disjoint is True
    assert loaded_msgpack.groupings[0].exhaustive is True
    assert validate(loaded_msgpack) is True

    # 3. CSV roundtrip
    csv_path = tmp_path / "test.bopp.csv"
    save_bopp_csv(ann, csv_path)
    loaded_csv = load_bopp_csv(csv_path)
    assert loaded_csv.extent.disjoint is True
    assert loaded_csv.extent.exhaustive is True
    assert loaded_csv.extent.min_time == 0.0
    assert loaded_csv.extent.max_time == 5.0
    assert loaded_csv.groupings[0].disjoint is True
    assert loaded_csv.groupings[0].exhaustive is True
    assert validate(loaded_csv) is True


# =====================================================================
# 5. Coverage and Edge Cases
# =====================================================================


def test_grouping_without_extent():
    with pytest.raises(BoppValidationError, match="has no extent"):
        create(
            media_id="audio:1",
            payload_type="tag",
            value=["a"],
            groupings=[{"name": "g", "ids": [0], "disjoint": True}],
        )
    with pytest.raises(BoppValidationError, match="has no extent"):
        create(
            media_id="audio:1",
            payload_type="tag",
            value=["a"],
            groupings=[{"name": "g", "ids": [0], "exhaustive": True}],
        )


def test_infer_and_get_bounds_edge_cases():
    import msgspec

    from bopp.models.v1.extent.time import Time
    from bopp.models.v1.extent.time_frequency_box import TimeFrequencyBoxExtent
    from bopp.models.v1.extent.time_uniform import UniformTimeExtent

    # None and UNSET inputs
    assert infer_extent_bounds(None) == {}
    assert infer_extent_bounds(msgspec.UNSET) == {}
    assert get_extent_bounds(None) == {}
    assert get_extent_bounds(msgspec.UNSET) == {}

    # Annotation with no extent
    ann_no_extent = create(media_id="audio:1", payload_type="tag", value=["a"])
    assert infer_extent_bounds(ann_no_extent) == {}
    assert get_extent_bounds(ann_no_extent) == {}

    # Annotation with payload length for time_uniform
    ann_with_p = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_uniform",
        value=["a", "b"],
        start_time=0.0,
        sampling_period=1.0,
    )
    assert infer_extent_bounds(ann_with_p) == {"time": (0.0, 1.0)}

    # Empty point extent
    empty_time = Time(time=[])
    assert infer_extent_bounds(empty_time) == {"time": (None, None)}
    # Point extent with subset indices
    time_pts = Time(time=[1.0, 5.0])
    assert infer_extent_bounds(time_pts, indices=[0]) == {"time": (1.0, 1.0)}

    # time_frequency_box
    tf_box = TimeFrequencyBoxExtent(
        time=[1.0, 3.0],
        duration=[2.0, 1.0],
        freq_min=[100.0, 200.0],
        freq_max=[500.0, 800.0],
    )
    assert infer_extent_bounds(tf_box) == {
        "time": (1.0, 4.0),
        "frequency": (100.0, 800.0),
    }
    assert infer_extent_bounds(tf_box, indices=[0]) == {
        "time": (1.0, 3.0),
        "frequency": (100.0, 500.0),
    }

    empty_tf = TimeFrequencyBoxExtent(time=[], duration=[], freq_min=[], freq_max=[])
    assert infer_extent_bounds(empty_tf) == {
        "time": (None, None),
        "frequency": (None, None),
    }

    # time_uniform
    uni = UniformTimeExtent(start_time=10.0, sampling_period=2.0)
    assert infer_extent_bounds(uni, N=0) == {"time": (None, None)}
    assert infer_extent_bounds(uni, N=1) == {"time": (10.0, 10.0)}
    assert infer_extent_bounds(uni, N=4) == {"time": (10.0, 16.0)}
    assert infer_extent_bounds(uni, indices=[0, 1]) == {"time": (10.0, 12.0)}


def test_union_area_2d_and_geometry():
    from bopp.util import _union_area_2d

    # Rectangles completely outside bounds -> clipped list is empty -> returns 0.0
    area = _union_area_2d([(10.0, 20.0, 10.0, 20.0)], (0.0, 5.0, 0.0, 5.0))
    assert area == 0.0

    # Non-overlapping horizontal strip
    area = _union_area_2d(
        [(0.0, 2.0, 0.0, 5.0), (3.0, 5.0, 0.0, 5.0)],
        (0.0, 5.0, 0.0, 5.0),
    )
    assert area == 20.0

    # Vertical gap between boxes (causes empty x_intervals slice)
    area_gap = _union_area_2d(
        [(0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 2.0, 3.0)],
        (0.0, 1.0, 0.0, 3.0),
    )
    assert area_gap == 2.0


def test_is_disjoint_edge_cases():
    import msgspec

    from bopp.models.v1.grouping import Grouping
    from bopp.util import _is_disjoint_indices

    assert is_disjoint(None) is True
    assert is_disjoint(msgspec.UNSET) is True

    class Dummy(msgspec.Struct, tag="dummy"):
        pass

    assert _is_disjoint_indices(Dummy(), None) is True

    # Degenerate interval collisions
    ann_deg = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[2.0, 2.0],
        duration=[0.0, 0.0],
        value=["a", "b"],
        disjoint=False,
    )
    assert is_disjoint(ann_deg) is False

    # 1D Point with indices
    ann_pt = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 2.0, 1.0],
        value=["a", "b", "c"],
        disjoint=False,
    )
    assert is_disjoint(ann_pt) is False
    assert (
        is_disjoint(ann_pt, grouping=Grouping(name="g", ids=[0, 1, 2]), group_id=0)
        is True
    )

    # Multi-dimensional point (Cartesian) with indices
    ann_cart = create(
        media_id="spatial:1",
        payload_type="tag",
        extent_type="cartesian",
        x=[0.0, 1.0, 0.0],
        y=[0.0, 1.0, 0.0],
        z=[0.0, 1.0, 0.0],
        value=["a", "b", "c"],
        disjoint=False,
    )
    assert is_disjoint(ann_cart) is False

    # 2D Box: time_frequency_box disjointness
    ann_tf_disj = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_frequency_box",
        time=[0.0, 2.0],
        duration=[1.0, 1.0],
        freq_min=[100.0, 100.0],
        freq_max=[500.0, 500.0],
        value=["b1", "b2"],
        disjoint=True,
    )
    assert is_disjoint(ann_tf_disj) is True

    # 0 or 1 box is always disjoint
    ann_tf_single = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_frequency_box",
        time=[0.0],
        duration=[1.0],
        freq_min=[100.0],
        freq_max=[500.0],
        value=["b1"],
        disjoint=True,
    )
    assert is_disjoint(ann_tf_single) is True

    # Overlapping time_frequency_box
    with pytest.raises(BoppValidationError, match="marked disjoint=True"):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_frequency_box",
            time=[0.0, 1.0],
            duration=[2.0, 2.0],
            freq_min=[100.0, 200.0],
            freq_max=[400.0, 500.0],
            value=["b1", "b2"],
            disjoint=True,
        )

    # spherical_time_interval disjointness
    ann_sph = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="spherical_time_interval",
        azimuth=[0.0, 1.0],
        elevation=[0.0, 1.0],
        radius=[1.0, 1.0],
        time=[0.0, 0.5],
        duration=[1.0, 1.0],
        value=["s1", "s2"],
        disjoint=True,
    )
    assert is_disjoint(ann_sph) is True

    # spherical_time_interval collision at same position
    with pytest.raises(BoppValidationError, match="marked disjoint=True"):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="spherical_time_interval",
            azimuth=[0.0, 0.0],
            elevation=[0.0, 0.0],
            radius=[1.0, 1.0],
            time=[0.0, 0.5],
            duration=[1.0, 1.0],
            value=["s1", "s2"],
            disjoint=True,
        )

    # Grouping with ids=None
    g_none = Grouping(name="g", ids=None)
    assert is_disjoint(ann_tf_disj, grouping=g_none) is True

    # Specific group_id evaluation
    g_multi = Grouping(name="g", ids=[0, 1, 0])
    ann_multi = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time",
        time=[1.0, 1.0, 1.0],
        value=["a", "b", "c"],
        disjoint=False,
    )
    assert is_disjoint(ann_multi, grouping=g_multi, group_id=1) is True
    assert is_disjoint(ann_multi, grouping=g_multi, group_id=0) is False

    # cartesian with indices
    assert (
        is_disjoint(ann_cart, grouping=Grouping(name="g", ids=[0, 1, 2]), group_id=0)
        is True
    )

    # pixel_box with indices
    ann_pb_multi = create(
        media_id="img:1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0, 0.0],
        width=[2.0, 2.0],
        y=[0.0, 0.0],
        height=[2.0, 2.0],
        value=["a", "b"],
        disjoint=False,
    )
    assert (
        is_disjoint(ann_pb_multi, grouping=Grouping(name="g", ids=[0, 1]), group_id=0)
        is True
    )

    # time_frequency_box with indices
    ann_tf_multi = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_frequency_box",
        time=[0.0, 0.0],
        duration=[2.0, 2.0],
        freq_min=[100.0, 100.0],
        freq_max=[500.0, 500.0],
        value=["a", "b"],
        disjoint=False,
    )
    assert (
        is_disjoint(ann_tf_multi, grouping=Grouping(name="g", ids=[0, 1]), group_id=0)
        is True
    )

    # cartesian_time_interval with indices
    ann_spt_multi = create(
        media_id="spat:1",
        payload_type="tag",
        extent_type="cartesian_time_interval",
        x=[0.0, 0.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        time=[0.0, 0.0],
        duration=[2.0, 2.0],
        value=["a", "b"],
        disjoint=False,
    )
    assert (
        is_disjoint(ann_spt_multi, grouping=Grouping(name="g", ids=[0, 1]), group_id=0)
        is True
    )


def test_is_exhaustive_edge_cases():
    import msgspec

    from bopp.models.v1.grouping import Grouping
    from bopp.util import _is_exhaustive_indices

    assert is_exhaustive(None) is False
    assert is_exhaustive(msgspec.UNSET) is False

    class Dummy(msgspec.Struct, tag="dummy"):
        pass

    assert _is_exhaustive_indices(Dummy(), indices=None, bounds={}, N=0) is True

    # time_uniform exhaustiveness
    ann_uni = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_uniform",
        start_time=0.0,
        sampling_period=1.0,
        value=["a", "b", "c"],
        min_time=0.0,
        max_time=2.0,
        exhaustive=True,
    )
    assert is_exhaustive(ann_uni) is True

    # time_uniform bounds not met
    with pytest.raises(BoppValidationError, match="marked exhaustive=True"):
        create(
            media_id="audio:1",
            payload_type="tag",
            extent_type="time_uniform",
            start_time=0.0,
            sampling_period=1.0,
            value=["a", "b", "c"],
            min_time=0.0,
            max_time=5.0,
            exhaustive=True,
        )

    # 1D Interval b_min > b_max or b_min == b_max
    ann_ti = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[1.0],
        duration=[2.0],
        value=["a"],
    )
    assert is_exhaustive(ann_ti, bounds={"time": (5.0, 2.0)}) is False
    assert is_exhaustive(ann_ti, bounds={"time": (2.0, 2.0)}) is True

    # text discrete point bounds
    ann_txt = create(
        media_id="doc:1",
        payload_type="tag",
        extent_type="text",
        offset=[0, 1, 2],
        value=["a", "b", "c"],
    )
    assert is_exhaustive(ann_txt, bounds={"offset": (5, 2)}) is False
    assert is_exhaustive(ann_txt, bounds={"offset": (1, 1)}) is True
    assert is_exhaustive(ann_txt, bounds={"offset": (10, 10)}) is False

    # 2D Box time_frequency_box exhaustiveness
    ann_tf_box = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_frequency_box",
        time=[0.0],
        duration=[2.0],
        freq_min=[100.0],
        freq_max=[500.0],
        value=["a"],
        min_time=0.0,
        max_time=2.0,
        min_freq=100.0,
        max_freq=500.0,
        exhaustive=True,
    )
    assert is_exhaustive(ann_tf_box) is True

    # pixel_box b_min == b_max
    ann_pb = create(
        media_id="img:1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0],
        width=[2.0],
        y=[0.0],
        height=[2.0],
        value=["a"],
    )
    assert is_exhaustive(ann_pb, bounds={"x": (0.0, 0.0), "y": (0.0, 0.0)}) is True
    assert is_exhaustive(ann_pb, bounds={"x": (5.0, 2.0), "y": (0.0, 2.0)}) is False

    # spatiotemporal intervals exhaustiveness
    ann_spt = create(
        media_id="spat:1",
        payload_type="tag",
        extent_type="cartesian_time_interval",
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        z=[0.0, 1.0],
        time=[0.0, 2.0],
        duration=[2.0, 2.0],
        value=["a", "b"],
        min_time=0.0,
        max_time=4.0,
        exhaustive=True,
    )
    assert is_exhaustive(ann_spt) is True
    assert is_exhaustive(ann_spt, bounds={"time": (5.0, 2.0)}) is False
    assert is_exhaustive(ann_spt, bounds={"time": (2.0, 2.0)}) is True

    # spatiotemporal with gap in time
    with pytest.raises(BoppValidationError, match="marked exhaustive=True"):
        create(
            media_id="spat:1",
            payload_type="tag",
            extent_type="cartesian_time_interval",
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            z=[0.0, 1.0],
            time=[0.0, 3.0],
            duration=[2.0, 1.0],
            value=["a", "b"],
            min_time=0.0,
            max_time=4.0,
            exhaustive=True,
        )

    # Continuous points bounds
    ann_pt_cont = create(
        media_id="spat:1",
        payload_type="tag",
        extent_type="cartesian",
        x=[0.0, 5.0],
        y=[0.0, 5.0],
        z=[0.0, 5.0],
        value=["a", "b"],
    )
    assert (
        is_exhaustive(
            ann_pt_cont, bounds={"x": (5.0, 2.0), "y": (0.0, 5.0), "z": (0.0, 5.0)}
        )
        is False
    )
    assert (
        is_exhaustive(
            ann_pt_cont, bounds={"x": (2.0, 2.0), "y": (0.0, 5.0), "z": (0.0, 5.0)}
        )
        is True
    )

    # Grouping exhaustiveness edge cases: ids is None, group_id, negative only
    g_none = Grouping(name="g", ids=None)
    assert is_exhaustive(ann_ti, grouping=g_none) is False

    g_neg = Grouping(name="g", ids=[-1, -1])
    assert is_exhaustive(ann_ti, grouping=g_neg) is False

    g_spec = Grouping(name="g", ids=[0, 1])
    ann_g_spec = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_interval",
        time=[0.0, 0.0],
        duration=[5.0, 2.0],
        value=["a", "b"],
        min_time=0.0,
        max_time=5.0,
    )
    assert is_exhaustive(ann_g_spec, grouping=g_spec, group_id=0) is True
    assert is_exhaustive(ann_g_spec, grouping=g_spec, group_id=1) is False

    ann_pb_multi = create(
        media_id="img:1",
        payload_type="tag",
        extent_type="pixel_box",
        x=[0.0, 0.0],
        width=[2.0, 2.0],
        y=[0.0, 0.0],
        height=[2.0, 2.0],
        value=["a", "b"],
        disjoint=False,
    )
    assert (
        is_exhaustive(
            ann_pb_multi,
            grouping=Grouping(name="g", ids=[0, 1]),
            group_id=0,
            bounds={"x": (0.0, 2.0), "y": (0.0, 2.0)},
        )
        is True
    )

    ann_tf_multi = create(
        media_id="audio:1",
        payload_type="tag",
        extent_type="time_frequency_box",
        time=[0.0, 0.0],
        duration=[2.0, 2.0],
        freq_min=[100.0, 100.0],
        freq_max=[500.0, 500.0],
        value=["a", "b"],
        disjoint=False,
    )
    assert (
        is_exhaustive(
            ann_tf_multi,
            grouping=Grouping(name="g", ids=[0, 1]),
            group_id=0,
            bounds={"time": (0.0, 2.0), "frequency": (100.0, 500.0)},
        )
        is True
    )

    ann_spt_multi = create(
        media_id="spat:1",
        payload_type="tag",
        extent_type="cartesian_time_interval",
        x=[0.0, 0.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        time=[0.0, 0.0],
        duration=[2.0, 2.0],
        value=["a", "b"],
        disjoint=False,
    )
    assert (
        is_exhaustive(
            ann_spt_multi,
            grouping=Grouping(name="g", ids=[0, 1]),
            group_id=0,
            bounds={"time": (0.0, 2.0)},
        )
        is True
    )
    # spatiotemporal with start gap and end gap
    ann_spt_gap = create(
        media_id="spat:1",
        payload_type="tag",
        extent_type="cartesian_time_interval",
        x=[0.0],
        y=[0.0],
        z=[0.0],
        time=[1.0],
        duration=[2.0],
        value=["a"],
    )
    assert is_exhaustive(ann_spt_gap, bounds={"time": (0.0, 3.0)}) is False
    assert is_exhaustive(ann_spt_gap, bounds={"time": (1.0, 4.0)}) is False

    # continuous points with indices and bounds
    ann_cart = create(
        media_id="spat:1",
        payload_type="tag",
        extent_type="cartesian",
        x=[0.0, 1.0, 0.0],
        y=[0.0, 1.0, 0.0],
        z=[0.0, 1.0, 0.0],
        value=["a", "b", "c"],
        disjoint=False,
    )
    assert (
        is_exhaustive(
            ann_cart,
            grouping=Grouping(name="g", ids=[0, 1, 2]),
            group_id=0,
            bounds={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        )
        is True
    )

    # text with indices and bounds
    assert (
        is_exhaustive(
            ann_txt,
            grouping=Grouping(name="g", ids=[0, 1, 2]),
            group_id=0,
            bounds={"offset": (0, 0)},
        )
        is True
    )

    # None bounds tests via _is_exhaustive_indices
    assert (
        _is_exhaustive_indices(
            ann_uni.extent, indices=None, bounds={"time": (None, None)}, N=3
        )
        is True
    )
    assert (
        _is_exhaustive_indices(
            ann_ti.extent, indices=None, bounds={"time": (None, None)}, N=1
        )
        is True
    )
    assert (
        _is_exhaustive_indices(
            ann_txt.extent, indices=None, bounds={"offset": (None, None)}, N=3
        )
        is True
    )
    assert (
        _is_exhaustive_indices(
            ann_pt_cont.extent,
            indices=None,
            bounds={"x": (None, None), "y": (None, None), "z": (None, None)},
            N=2,
        )
        is True
    )
