from unittest.mock import MagicMock, patch

import msgspec
import pytest

from bopp.core import compute_annotation_id, create, validate_annotation_id
from bopp.exceptions import BoppRegistryError
from bopp.extensions import (
    ExtensionRegistry,
    get_extensions,
    reset_extensions,
    update_extensions,
)
from bopp.io import (
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    resolve_extensions,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)


class CustomItem(msgspec.Struct, forbid_unknown_fields=True):
    name: str
    score: float


def test_extension_registry_mapping():
    reg = ExtensionRegistry()
    reg["test_key"] = CustomItem

    assert len(reg) == 1
    assert "test_key" in reg
    assert reg["test_key"] is CustomItem
    assert list(reg) == ["test_key"]

    del reg["test_key"]
    assert len(reg) == 0
    assert "test_key" not in reg


def test_lazy_entry_point_loading():
    mock_ep = MagicMock()
    mock_ep.name = "example.custom"
    mock_ep.value = "dummy_module:CustomItem"

    reg = ExtensionRegistry()
    reg.register_entry("example.custom", mock_ep)

    # Resolution should not occur upon registration
    assert "example.custom" not in reg._resolved

    with patch("bopp.extensions._lazy_loader.load", return_value=CustomItem) as mock_load:
        loaded = reg["example.custom"]
        mock_load.assert_called_once_with("dummy_module:CustomItem")

    assert loaded is CustomItem
    # Once resolved, repeated access uses cached value without calling loader again
    assert reg["example.custom"] is CustomItem
    assert "example.custom" in reg._resolved


def test_entry_point_fallback_load():
    mock_ep = MagicMock()
    mock_ep.name = "example.fallback"
    mock_ep.value = "nonexistent.module:SomeClass"
    mock_ep.load.return_value = CustomItem

    reg = ExtensionRegistry()
    reg.register_entry("example.fallback", mock_ep)

    mock_ep.load.assert_not_called()
    loaded = reg["example.fallback"]
    assert loaded is CustomItem
    mock_ep.load.assert_called_once()


def test_update_extensions_conflict_warning():
    ep1 = MagicMock()
    ep1.name = "conflict.schema"
    ep1.value = "pkg_a:SchemaA"

    ep2 = MagicMock()
    ep2.name = "conflict.schema"
    ep2.value = "pkg_b:SchemaB"

    with patch("importlib.metadata.entry_points", return_value=[ep1, ep2]):
        reset_extensions()
        with pytest.warns(UserWarning, match="Conflict for extension 'conflict.schema'"):
            update_extensions()

    extensions = get_extensions()
    assert "conflict.schema" in extensions
    # Keeps first registration
    assert extensions._raw_entries["conflict.schema"] == ep1


def test_resolve_extensions_non_extension_payload():
    ann = create(
        bopp_version="1.0",
        media_id="track:onset_test",
        payload_kind="onset",
        extent_kind="time",
        time=[0.1],
        value=[1],
    )
    # Should be a no-op without error
    resolve_extensions(ann)
    assert ann.payload.value == [1]


def test_resolve_extensions_missing_schema_warning():
    ann = create(
        bopp_version="1.0",
        media_id="track:ext_test",
        payload_kind="ext",
        ext_schema="unregistered.schema",
        value=[{"name": "foo", "score": 1.0}],
        resolve_ext=False,
    )

    with pytest.warns(UserWarning, match="No extension registered for schema"):
        resolve_extensions(ann, allow_missing=True)


def test_resolve_extensions_missing_schema_error():
    ann = create(
        bopp_version="1.0",
        media_id="track:ext_test",
        payload_kind="ext",
        ext_schema="unregistered.schema",
        value=[{"name": "foo", "score": 1.0}],
        resolve_ext=False,
    )

    with pytest.raises(BoppRegistryError, match="Extension schema 'unregistered.schema' not found"):
        resolve_extensions(ann, allow_missing=False)


def test_resolve_extensions_successful_conversion():
    exts = get_extensions()
    exts["org.test.custom"] = CustomItem

    try:
        ann = create(
            bopp_version="1.0",
            media_id="track:ext_test",
            payload_kind="ext",
            ext_schema="org.test.custom",
            value=[{"name": "test", "score": 42.0}],
            resolve_ext=False,
        )

        resolve_extensions(ann)

        assert isinstance(ann.payload.value[0], CustomItem)
        assert ann.payload.value[0].name == "test"
        assert ann.payload.value[0].score == 42.0
    finally:
        del exts["org.test.custom"]


def test_resolve_extensions_validation_error():
    exts = get_extensions()
    exts["org.test.custom"] = CustomItem

    try:
        ann = create(
            bopp_version="1.0",
            media_id="track:ext_test",
            payload_kind="ext",
            ext_schema="org.test.custom",
            value=[{"name": "test", "score": "not-a-float"}],
            resolve_ext=False,
        )

        with pytest.raises(msgspec.ValidationError):
            resolve_extensions(ann)
    finally:
        del exts["org.test.custom"]


def test_resolve_extensions_strict_coercion_rejected():
    """Verify that strict=True conversion rejects string to float coercion."""
    exts = get_extensions()
    exts["org.test.custom"] = CustomItem

    try:
        ann = create(
            bopp_version="1.0",
            media_id="track:ext_test",
            payload_kind="ext",
            ext_schema="org.test.custom",
            value=[{"name": "test", "score": "12.34"}],
            resolve_ext=False,
        )
        with pytest.raises(msgspec.ValidationError):
            resolve_extensions(ann)
    finally:
        del exts["org.test.custom"]


def test_annotation_id_invariance_with_extension_resolution():
    """Verify that annotation ID matches identically with raw dict vs resolved struct."""
    exts = get_extensions()
    exts["org.test.custom"] = CustomItem

    try:
        raw_items = [{"name": "item1", "score": 10.5}, {"name": "item2", "score": 20.0}]

        ann_unresolved = create(
            bopp_version="1.0",
            media_id="track:invariance_test",
            payload_kind="ext",
            ext_schema="org.test.custom",
            value=raw_items,
            resolve_ext=False,
        )

        ann_resolved = create(
            bopp_version="1.0",
            media_id="track:invariance_test",
            payload_kind="ext",
            ext_schema="org.test.custom",
            value=raw_items,
            resolve_ext=True,
        )

        id_unresolved = compute_annotation_id(ann_unresolved)
        id_resolved = compute_annotation_id(ann_resolved)

        assert id_unresolved == id_resolved
        assert validate_annotation_id(ann_unresolved)
        assert validate_annotation_id(ann_resolved)
    finally:
        del exts["org.test.custom"]


def test_create_with_resolve_ext():
    exts = get_extensions()
    exts["org.test.custom"] = CustomItem

    try:
        ann = create(
            bopp_version="1.0",
            media_id="track:ext_test",
            payload_kind="ext",
            ext_schema="org.test.custom",
            value=[{"name": "item1", "score": 1.5}],
            resolve_ext=True,
        )
        assert isinstance(ann.payload.value[0], CustomItem)

        ann_unresolved = create(
            bopp_version="1.0",
            media_id="track:ext_test",
            payload_kind="ext",
            ext_schema="org.test.custom",
            value=[{"name": "item1", "score": 1.5}],
            resolve_ext=False,
        )
        assert isinstance(ann_unresolved.payload.value[0], dict)
    finally:
        del exts["org.test.custom"]


def test_io_loaders_resolve_ext(tmp_path):
    exts = get_extensions()
    exts["org.test.custom"] = CustomItem

    try:
        ann = create(
            bopp_version="1.0",
            media_id="track:ext_io",
            payload_kind="ext",
            ext_schema="org.test.custom",
            value=[{"name": "item1", "score": 2.5}],
            resolve_ext=False,
        )

        json_path = tmp_path / "ext.json"
        msgpack_path = tmp_path / "ext.msgpack"
        csv_path = tmp_path / "ext.csv"

        save_bopp_json(ann, json_path)
        save_bopp_msgpack(ann, msgpack_path)
        save_bopp_csv(ann, csv_path)

        # JSON loader
        loaded_json_resolved = load_bopp_json(json_path, resolve_ext=True)
        assert isinstance(loaded_json_resolved.payload.value[0], CustomItem)

        loaded_json_unresolved = load_bopp_json(json_path, resolve_ext=False)
        assert isinstance(loaded_json_unresolved.payload.value[0], dict)

        # MsgPack loader
        loaded_msgpack_resolved = load_bopp_msgpack(msgpack_path, resolve_ext=True)
        assert isinstance(loaded_msgpack_resolved.payload.value[0], CustomItem)

        loaded_msgpack_unresolved = load_bopp_msgpack(msgpack_path, resolve_ext=False)
        assert isinstance(loaded_msgpack_unresolved.payload.value[0], dict)

        # CSV loader
        loaded_csv_resolved = load_bopp_csv(csv_path, resolve_ext=True)
        assert isinstance(loaded_csv_resolved.payload.value[0], CustomItem)

        loaded_csv_unresolved = load_bopp_csv(csv_path, resolve_ext=False)
        assert isinstance(loaded_csv_unresolved.payload.value[0], dict)
    finally:
        del exts["org.test.custom"]
