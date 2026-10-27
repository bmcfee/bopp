import numpy as np
import pytest

from bopp.core import create, validate
from bopp.evaluation import _extract_events, _extract_intervals, _extract_labels
from bopp.exceptions import BoppArgumentError, BoppArrayError, BoppValidationError
from bopp.io import load_bopp_csv, save_bopp_csv
from bopp.registries.v1 import EXTENT_TYPE_REGISTRY
from bopp.transforms import filter_by, to_times, trim
from bopp.util import from_dataframe, to_dataframe

NEW_EXTENT_FIXTURES = [
    (
        "pixel",
        {"x": [10.5, 20.0, 35.2], "y": [5.0, 15.5, 40.0]},
    ),
    (
        "text",
        {"offset": [0, 15, 42]},
    ),
    (
        "text_interval",
        {"offset": [0, 15, 42], "length": [5, 10, 8]},
    ),
    (
        "spherical",
        {
            "azimuth": [-180.0, 0.0, 179.9],
            "elevation": [-90.0, 0.0, 90.0],
            "radius": [1.0, 2.5, 10.0],
        },
    ),
    (
        "spherical_time",
        {
            "time": [0.1, 0.5, 1.2],
            "azimuth": [-90.0, 45.0, 120.0],
            "elevation": [-30.0, 60.0, -10.0],
            "radius": [2.0, 3.0, 5.0],
        },
    ),
    (
        "spherical_time_interval",
        {
            "time": [0.1, 0.5, 1.2],
            "duration": [0.2, 0.4, 0.5],
            "azimuth": [-90.0, 45.0, 120.0],
            "elevation": [-30.0, 60.0, -10.0],
            "radius": [2.0, 3.0, 5.0],
        },
    ),
    (
        "cartesian",
        {"x": [0.0, 1.0, -2.5], "y": [2.0, 3.0, 0.5], "z": [-1.0, 4.0, 10.2]},
    ),
    (
        "cartesian_time",
        {
            "time": [0.0, 1.0, 2.0],
            "x": [0.0, 1.0, -2.5],
            "y": [2.0, 3.0, 0.5],
            "z": [-1.0, 4.0, 10.2],
        },
    ),
    (
        "cartesian_time_interval",
        {
            "time": [0.0, 1.0, 2.0],
            "duration": [0.5, 0.8, 0.4],
            "x": [0.0, 1.0, -2.5],
            "y": [2.0, 3.0, 0.5],
            "z": [-1.0, 4.0, 10.2],
        },
    ),
    (
        "time_uniform",
        {"start_time": 0.0, "sampling_period": 0.05},
    ),
]


@pytest.mark.parametrize("extent_kind, extent_kwargs", NEW_EXTENT_FIXTURES)
def test_create_and_validate_new_extents(extent_kind, extent_kwargs):
    cls = EXTENT_TYPE_REGISTRY[extent_kind]
    assert cls is not None

    ann = create(
        media_id=f"media:{extent_kind}",
        payload_kind="tag",
        extent_kind=extent_kind,
        value=["obs_1", "obs_2", "obs_3"],
        **extent_kwargs,
    )
    assert validate(ann) is True
    assert isinstance(ann.extent, cls)


def test_pixel_validation():
    # Valid pixel
    ann = create(
        media_id="audio:pixel",
        payload_kind="tag",
        extent_kind="pixel",
        x=[1.0, 2.0],
        y=[3.0, 4.0],
        value=["p1", "p2"],
    )
    assert validate(ann) is True

    # Missing y raises TypeError at construction
    with pytest.raises(TypeError):
        create(
            media_id="audio:pixel",
            payload_kind="tag",
            extent_kind="pixel",
            x=[1.0],
            value=["p1"],
        )


def test_text_and_text_interval_constraints():
    # Valid text
    ann_text = create(
        media_id="doc:1",
        payload_kind="tag",
        extent_kind="text",
        offset=[0, 10, 25],
        value=["token1", "token2", "token3"],
    )
    assert validate(ann_text) is True

    # Invalid negative offset rejects on validation
    bad_text = create(
        media_id="doc:bad_text",
        payload_kind="tag",
        extent_kind="text",
        offset=[-1],
        value=["bad"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_text)

    # Valid text_interval
    ann_ti = create(
        media_id="doc:2",
        payload_kind="tag",
        extent_kind="text_interval",
        offset=[0, 10],
        length=[5, 8],
        value=["span1", "span2"],
    )
    assert validate(ann_ti) is True

    # Invalid negative length rejects on validation
    bad_ti = create(
        media_id="doc:bad_ti",
        payload_kind="tag",
        extent_kind="text_interval",
        offset=[0],
        length=[-2],
        value=["bad"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_ti)


def test_spherical_range_constraints():
    # Valid bounds
    ann_valid = create(
        media_id="audio:spherical",
        payload_kind="tag",
        extent_kind="spherical",
        azimuth=[-180.0, 0.0, 179.999],
        elevation=[-90.0, 0.0, 90.0],
        radius=[0.001, 1.0, 100.0],
        value=["src1", "src2", "src3"],
    )
    assert validate(ann_valid) is True

    # Azimuth < -180.0
    bad_az1 = create(
        media_id="audio:sph",
        payload_kind="tag",
        extent_kind="spherical",
        azimuth=[-180.1],
        elevation=[0.0],
        radius=[1.0],
        value=["v"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_az1)

    # Azimuth >= 180.0
    bad_az2 = create(
        media_id="audio:sph",
        payload_kind="tag",
        extent_kind="spherical",
        azimuth=[180.0],
        elevation=[0.0],
        radius=[1.0],
        value=["v"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_az2)

    # Elevation < -90.0
    bad_el1 = create(
        media_id="audio:sph",
        payload_kind="tag",
        extent_kind="spherical",
        azimuth=[0.0],
        elevation=[-90.1],
        radius=[1.0],
        value=["v"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_el1)

    # Elevation > 90.0
    bad_el2 = create(
        media_id="audio:sph",
        payload_kind="tag",
        extent_kind="spherical",
        azimuth=[0.0],
        elevation=[90.1],
        radius=[1.0],
        value=["v"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_el2)

    # Radius <= 0.0
    bad_r0 = create(
        media_id="audio:sph",
        payload_kind="tag",
        extent_kind="spherical",
        azimuth=[0.0],
        elevation=[0.0],
        radius=[0.0],
        value=["v"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_r0)


def test_time_uniform_constraints():
    # Valid time_uniform
    ann = create(
        media_id="audio:uniform",
        payload_kind="tag",
        extent_kind="time_uniform",
        start_time=0.0,
        sampling_period=0.01,
        value=["a", "b", "c"],
    )
    assert validate(ann) is True
    assert ann.extent.start_time == 0.0
    assert ann.extent.sampling_period == 0.01

    # sampling_period <= 0 rejects
    bad_period0 = create(
        media_id="audio:uniform_bad",
        payload_kind="tag",
        extent_kind="time_uniform",
        start_time=0.0,
        sampling_period=0.0,
        value=["a"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_period0)

    bad_period_neg = create(
        media_id="audio:uniform_bad",
        payload_kind="tag",
        extent_kind="time_uniform",
        start_time=1.0,
        sampling_period=-0.05,
        value=["a"],
    )
    with pytest.raises(BoppValidationError):
        validate(bad_period_neg)


def test_array_length_mismatch_raises():
    # pixel mismatch
    with pytest.raises(BoppArrayError):
        create(
            media_id="audio:err",
            payload_kind="tag",
            extent_kind="pixel",
            x=[1.0, 2.0],
            y=[1.0],
            value=["a", "b"],
        )

    # cartesian mismatch
    with pytest.raises(BoppArrayError):
        create(
            media_id="audio:err",
            payload_kind="tag",
            extent_kind="cartesian",
            x=[1.0, 2.0],
            y=[1.0, 2.0],
            z=[1.0],
            value=["a", "b"],
        )

    # spherical mismatch
    with pytest.raises(BoppArrayError):
        create(
            media_id="audio:err",
            payload_kind="tag",
            extent_kind="spherical",
            azimuth=[0.0, 10.0],
            elevation=[0.0],
            radius=[1.0, 1.0],
            value=["a", "b"],
        )

    # text_interval mismatch
    with pytest.raises(BoppArrayError):
        create(
            media_id="audio:err",
            payload_kind="tag",
            extent_kind="text_interval",
            offset=[0, 10],
            length=[5],
            value=["a", "b"],
        )


@pytest.mark.parametrize("extent_kind, extent_kwargs", NEW_EXTENT_FIXTURES)
def test_dataframe_roundtrip_new_extents(extent_kind, extent_kwargs):
    ann = create(
        media_id=f"track:{extent_kind}_df",
        payload_kind="tag",
        extent_kind=extent_kind,
        value=["v1", "v2", "v3"],
        **extent_kwargs,
    )
    for backend in ("pandas", "polars"):
        ann_df = to_dataframe(ann, backend=backend)
        if extent_kind == "time_uniform":
            assert ann_df.header["extent"]["extent_type"] == "time_uniform"
            assert ann_df.header["extent"]["start_time"] == 0.0
            assert ann_df.header["extent"]["sampling_period"] == 0.05
        else:
            assert f"extent:{extent_kind}:" in "".join(ann_df.df.columns)

        reconstructed = from_dataframe(ann_df)
        assert reconstructed.extent == ann.extent
        assert reconstructed.payload.value == ann.payload.value
        assert validate(reconstructed) is True


@pytest.mark.parametrize("extent_kind, extent_kwargs", NEW_EXTENT_FIXTURES)
def test_csv_roundtrip_new_extents(tmp_path, extent_kind, extent_kwargs):
    ann = create(
        media_id=f"track:{extent_kind}_csv",
        payload_kind="tag",
        extent_kind=extent_kind,
        value=["v1", "v2", "v3"],
        **extent_kwargs,
    )
    csv_path = tmp_path / f"{extent_kind}.csv"
    save_bopp_csv(ann, csv_path)

    loaded = load_bopp_csv(csv_path)
    assert loaded.extent == ann.extent
    assert loaded.payload.value == ann.payload.value
    assert validate(loaded) is True


def test_trim_new_extents():
    # Trim pixel along x and y
    ann_pixel = create(
        media_id="track:pixel_trim",
        payload_kind="tag",
        extent_kind="pixel",
        x=[10.0, 20.0, 30.0, 40.0],
        y=[5.0, 15.0, 25.0, 35.0],
        value=["a", "b", "c", "d"],
    )
    trimmed_x = trim(ann_pixel, start=15.0, end=35.0, target_field="x")
    assert trimmed_x.extent.x == [20.0, 30.0]
    assert trimmed_x.extent.y == [15.0, 25.0]
    assert trimmed_x.payload.value == ["b", "c"]

    trimmed_y_reset = trim(
        ann_pixel, start=15.0, end=35.0, target_field="y", reset=True
    )
    assert trimmed_y_reset.extent.y == [0.0, 10.0, 20.0]
    assert trimmed_y_reset.payload.value == ["b", "c", "d"]

    # Trim text along offset
    ann_text = create(
        media_id="track:text_trim",
        payload_kind="tag",
        extent_kind="text",
        offset=[0, 10, 20, 30],
        value=["w0", "w1", "w2", "w3"],
    )
    trimmed_text = trim(ann_text, start=8, end=25)
    assert trimmed_text.extent.offset == [10, 20]
    assert trimmed_text.payload.value == ["w1", "w2"]

    # Trim text_interval
    ann_ti = create(
        media_id="track:ti_trim",
        payload_kind="tag",
        extent_kind="text_interval",
        offset=[0, 10, 20],
        length=[5, 8, 10],
        value=["span0", "span1", "span2"],
    )
    trimmed_ti = trim(ann_ti, start=5, end=25)
    # span0 [0, 5] overlaps at boundary -> clipped to [5, 5] (len 0) or span1 [10, 18] included
    assert "span1" in trimmed_ti.payload.value

    # Trim spherical along azimuth
    ann_sph = create(
        media_id="track:sph_trim",
        payload_kind="tag",
        extent_kind="spherical",
        azimuth=[-90.0, 0.0, 45.0, 135.0],
        elevation=[0.0, 10.0, -20.0, 30.0],
        radius=[1.0, 2.0, 3.0, 4.0],
        value=["s1", "s2", "s3", "s4"],
    )
    trimmed_sph = trim(ann_sph, start=-10.0, end=50.0, target_field="azimuth")
    assert trimmed_sph.extent.azimuth == [0.0, 45.0]
    assert trimmed_sph.payload.value == ["s2", "s3"]

    # Trim cartesian_time along default 'time'
    ann_ct = create(
        media_id="track:ct_trim",
        payload_kind="tag",
        extent_kind="cartesian_time",
        time=[0.0, 1.0, 2.0, 3.0],
        x=[0.0, 1.0, 2.0, 3.0],
        y=[0.0, 1.0, 2.0, 3.0],
        z=[0.0, 1.0, 2.0, 3.0],
        value=["c0", "c1", "c2", "c3"],
    )
    trimmed_ct = trim(ann_ct, start=0.5, end=2.5)
    assert trimmed_ct.extent.time == [1.0, 2.0]
    assert trimmed_ct.payload.value == ["c1", "c2"]

    # Trim time_uniform
    ann_tu = create(
        media_id="track:tu_trim",
        payload_kind="tag",
        extent_kind="time_uniform",
        start_time=0.0,
        sampling_period=0.1,
        value=["u0", "u1", "u2", "u3", "u4"],
    )
    # Timestamps are 0.0, 0.1, 0.2, 0.3, 0.4
    trimmed_tu = trim(ann_tu, start=0.15, end=0.35)
    assert trimmed_tu.extent.start_time == 0.2
    assert trimmed_tu.extent.sampling_period == 0.1
    assert trimmed_tu.payload.value == ["u2", "u3"]

    # Trim time_uniform with reset=True
    trimmed_tu_reset = trim(ann_tu, start=0.15, end=0.35, reset=True)
    assert np.isclose(trimmed_tu_reset.extent.start_time, 0.05)
    assert trimmed_tu_reset.payload.value == ["u2", "u3"]


def test_to_times_new_extents():
    # to_times on time_uniform
    ann_tu = create(
        media_id="track:tu_totimes",
        payload_kind="tag",
        extent_kind="time_uniform",
        start_time=1.0,
        sampling_period=0.5,
        value=["t1", "t2", "t3"],
    )
    # Timestamps: 1.0, 1.5, 2.0
    resampled = to_times(ann_tu, times=[1.1, 1.6, 2.1], method="previous")
    assert resampled.extent.time == [1.1, 1.6, 2.1]
    assert resampled.payload.value == ["t1", "t2", "t3"]

    # to_times with sample_rate on time_uniform
    resampled_rate = to_times(ann_tu, sample_rate=2.0)
    assert len(resampled_rate.extent.time) == 3

    # to_times on spherical_time
    ann_st = create(
        media_id="track:st_totimes",
        payload_kind="tag",
        extent_kind="spherical_time",
        time=[0.0, 2.0],
        azimuth=[0.0, 90.0],
        elevation=[0.0, 45.0],
        radius=[1.0, 2.0],
        value=["v1", "v2"],
    )
    resampled_st = to_times(ann_st, times=[0.5, 1.9], method="nearest")
    assert resampled_st.payload.value == ["v1", "v2"]

    # to_times on cartesian_time_interval
    ann_cti = create(
        media_id="track:cti_totimes",
        payload_kind="tag",
        extent_kind="cartesian_time_interval",
        time=[0.0, 2.0],
        duration=[1.0, 1.0],
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        z=[0.0, 1.0],
        value=["i1", "i2"],
    )
    resampled_cti = to_times(ann_cti, times=[0.5, 2.5])
    assert resampled_cti.payload.value == ["i1", "i2"]


def test_evaluation_helpers_new_extents():
    # _extract_events on cartesian_time
    ann_ct = create(
        media_id="a",
        payload_kind="tag",
        extent_kind="cartesian_time",
        time=[1.0, 2.5],
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        z=[0.0, 1.0],
        value=["a", "b"],
    )
    ev = _extract_events(ann_ct)
    np.testing.assert_array_equal(ev, [1.0, 2.5])

    # _extract_events on time_uniform
    ann_tu = create(
        media_id="a",
        payload_kind="tag",
        extent_kind="time_uniform",
        start_time=2.0,
        sampling_period=0.5,
        value=["a", "b", "c"],
    )
    ev_tu = _extract_events(ann_tu)
    np.testing.assert_allclose(ev_tu, [2.0, 2.5, 3.0])

    # _extract_intervals on spherical_time_interval
    ann_sti = create(
        media_id="a",
        payload_kind="tag",
        extent_kind="spherical_time_interval",
        time=[1.0, 4.0],
        duration=[2.0, 3.0],
        azimuth=[0.0, 10.0],
        elevation=[0.0, 10.0],
        radius=[1.0, 2.0],
        value=["a", "b"],
    )
    iv = _extract_intervals(ann_sti)
    np.testing.assert_allclose(iv, [[1.0, 3.0], [4.0, 7.0]])

    # _extract_labels on time_uniform with inferred intervals
    labels = _extract_labels(ann_tu, "value")
    assert labels == ["a", "b"]


def test_filter_by_and_errors_new_extents():
    # Filter by on cartesian extent
    ann_cart = create(
        media_id="a",
        payload_kind="tag",
        extent_kind="cartesian",
        x=[0.0, 5.0, 10.0],
        y=[0.0, 1.0, 2.0],
        z=[-1.0, 0.0, 1.0],
        value=["c0", "c1", "c2"],
    )
    filtered = filter_by(ann_cart, lambda x: x > 2.0, target="x", facet="extent")
    assert filtered.extent.x == [5.0, 10.0]
    assert filtered.payload.value == ["c1", "c2"]

    # Trim cartesian without specifying target_field raises BoppArgumentError
    with pytest.raises(BoppArgumentError, match="does not have a default target field"):
        trim(ann_cart, start=1.0)

    # Trim time_uniform with invalid target_field raises BoppArgumentError
    ann_tu = create(
        media_id="a",
        payload_kind="tag",
        extent_kind="time_uniform",
        start_time=0.0,
        sampling_period=0.1,
        value=["a", "b"],
    )
    with pytest.raises(
        BoppArgumentError,
        match="Unsupported target_field 'x' for extent tag 'time_uniform'",
    ):
        trim(ann_tu, start=0.0, target_field="x")
