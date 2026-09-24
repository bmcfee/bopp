from bopp.core import create
from bopp.io import (
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)
from bopp.util import _get_tag


def test_json_roundtrip(tmp_path):
    ann = create(
        media_id="track:json_test",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.5, 1.2],
        value=[1, 1, 1],
        sandbox={"user_note": "json_test", "flag": True},
    )
    file_path = tmp_path / "test.json"

    save_bopp_json(ann, file_path)
    loaded = load_bopp_json(file_path)

    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == _get_tag(ann.payload)
    assert getattr(loaded, "sandbox", None) == {"user_note": "json_test", "flag": True}


def test_msgpack_roundtrip(tmp_path):
    ann = create(
        media_id="track:msgpack_test",
        payload_kind="onset",
        extent_kind="time",
        time=[0.3, 0.7],
        value=[1, 1],
        sandbox={"user_note": "msgpack_test", "flag": False},
    )
    file_path = tmp_path / "test.msgpack"

    save_bopp_msgpack(ann, file_path)
    loaded = load_bopp_msgpack(file_path)

    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == _get_tag(ann.payload)
    assert getattr(loaded, "sandbox", None) == {"user_note": "msgpack_test", "flag": False}


def test_csv_roundtrip(tmp_path):
    ann = create(
        media_id="track:csv_test",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1, 0.4],
        value=[1, 1],
        sandbox={"user_note": "csv_test", "count": 10},
    )
    file_path = tmp_path / "test.csv"

    save_bopp_csv(ann, file_path)

    # Test direct loading into Annotation struct
    loaded = load_bopp_csv(file_path)
    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == _get_tag(ann.payload)
    assert getattr(loaded, "sandbox", None) == {"user_note": "csv_test", "count": 10}
