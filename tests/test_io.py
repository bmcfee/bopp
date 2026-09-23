import pandas as pd

from bopp.core import create
from bopp.io import (
    from_dataframe,
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    read_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
    to_csv,
)
from bopp.util import _get_tag


def test_json_roundtrip(tmp_path):
    ann = create(
        media_id="track:json_test",
        payload_kind="onset",
        extent_kind="timestamps",
        time=[0.1, 0.5, 1.2],
        value=[1, 1, 1],
    )
    file_path = tmp_path / "test.json"

    save_bopp_json(ann, file_path)
    loaded = load_bopp_json(file_path)

    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == _get_tag(ann.payload)


def test_msgpack_roundtrip(tmp_path):
    ann = create(
        media_id="track:msgpack_test",
        payload_kind="onset",
        extent_kind="timestamps",
        time=[0.3, 0.7],
        value=[1, 1],
    )
    file_path = tmp_path / "test.msgpack"

    save_bopp_msgpack(ann, file_path)
    loaded = load_bopp_msgpack(file_path)

    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == _get_tag(ann.payload)


def test_csv_roundtrip(tmp_path):
    ann = create(
        media_id="track:csv_test",
        payload_kind="onset",
        extent_kind="timestamps",
        time=[0.1, 0.4],
        value=[1, 1],
    )
    file_path = tmp_path / "test.csv"

    to_csv(ann, file_path)

    # Test read_bopp_csv returning DataFrame with attrs
    df = read_bopp_csv(file_path)
    assert isinstance(df, pd.DataFrame)
    assert df.attrs["media_id"] == "track:csv_test"

    # Test direct loading into Annotation struct
    loaded = load_bopp_csv(file_path)
    assert loaded.media_id == ann.media_id
    assert _get_tag(loaded.payload) == _get_tag(ann.payload)


def test_read_bopp_csv_list_evaluation(tmp_path):
    csv_content = (
        "# ---\n"
        'media_id = "track:list_test"\n'
        'bopp_version = "v1"\n'
        "# ---\n"
        "payload:chord:value\n"
        '"[\'C\', \'E\', \'G\']"\n'
    )
    file_path = tmp_path / "polyphonic.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    df = read_bopp_csv(file_path)
    assert isinstance(df["payload:chord:value"].iloc[0], list)
    assert df["payload:chord:value"].iloc[0] == ["C", "E", "G"]
