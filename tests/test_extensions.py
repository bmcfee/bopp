import types
import warnings
from types import SimpleNamespace
from typing import Union
from unittest.mock import MagicMock, patch

import msgspec
import pytest

import bopp.extensions as ext_mod
from bopp.core import compute_annotation_id, create, validate_annotation_id
from bopp.exceptions import BoppRegistryError
from bopp.extensions import (
    REGISTRY,
    ExtensionRegistry,
    get_extensions,
    reset_extensions,
    resolve_extensions,
    update_extensions,
)
from bopp.io import (
    load_bopp_csv,
    load_bopp_json,
    load_bopp_msgpack,
    save_bopp_csv,
    save_bopp_json,
    save_bopp_msgpack,
)


class CustomItem(msgspec.Struct, forbid_unknown_fields=True):
    name: str
    score: float


class AlternativeItem(msgspec.Struct, forbid_unknown_fields=True):
    code: int
    label: str


class TaggedItemA(msgspec.Struct, tag=True, forbid_unknown_fields=True):
    name: str
    score: float


class TaggedItemB(msgspec.Struct, tag=True, forbid_unknown_fields=True):
    code: int
    label: str


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
    mock_ep.load.return_value = CustomItem

    reg = ExtensionRegistry()
    reg.register_entry("example.custom", mock_ep)

    # Resolution should not occur upon registration
    assert "example.custom" not in reg._resolved
    mock_ep.load.assert_not_called()

    # Resolution occurs on first access
    loaded = reg["example.custom"]
    mock_ep.load.assert_called_once()
    assert loaded is CustomItem

    # Once resolved, repeated access uses cached value without calling load again
    assert reg["example.custom"] is CustomItem
    assert mock_ep.load.call_count == 1
    assert "example.custom" in reg._resolved


def test_string_spec_entry_point_loading():
    reg = ExtensionRegistry()
    reg.register_entry("example.string_spec", f"{__name__}:CustomItem")

    assert "example.string_spec" not in reg._resolved

    loaded = reg["example.string_spec"]
    assert loaded is CustomItem
    assert reg["example.string_spec"] is CustomItem
    assert "example.string_spec" in reg._resolved


def test_union_type_support():
    """Verify that types.UnionType (PEP 604) and typing.Union are supported as extension schemas."""
    target_union = TaggedItemA | TaggedItemB
    assert isinstance(target_union, types.UnionType)

    reg = ExtensionRegistry()
    reg.register_entry("example.union", f"{__name__}:TaggedItemA")
    reg["example.union"] = target_union

    assert reg["example.union"] == target_union
    assert "example.union" in reg

    # Also test typing.Union
    typing_union = Union[TaggedItemA, TaggedItemB]  # noqa: UP007
    reg["example.typing_union"] = typing_union
    assert reg["example.typing_union"] == typing_union

    # Test resolving payload with PEP 604 union type
    exts = get_extensions()
    exts["org.test.union"] = target_union
    try:
        ann = create(
            bopp_version="1.0",
            media_id="track:union_test",
            payload_kind="ext",
            ext_schema="org.test.union",
            value=[
                {"type": "TaggedItemA", "name": "foo", "score": 1.0},
                {"type": "TaggedItemB", "code": 42, "label": "answer"},
            ],
            resolve_ext=False,
        )
        resolve_extensions(ann)
        assert isinstance(ann.payload.value[0], TaggedItemA)
        assert ann.payload.value[0].name == "foo"
        assert isinstance(ann.payload.value[1], TaggedItemB)
        assert ann.payload.value[1].code == 42
    finally:
        del exts["org.test.union"]


def test_update_extensions_conflict_warning():
    ep1 = SimpleNamespace(
        name="conflict.schema", value="pkg_a:SchemaA", load=lambda: object
    )
    ep2 = SimpleNamespace(
        name="conflict.schema", value="pkg_b:SchemaB", load=lambda: object
    )
    valid_ep = SimpleNamespace(
        name="valid.schema",
        value=f"{__name__}:CustomItem",
        load=lambda: CustomItem,
    )

    try:
        reset_extensions()
        with patch(
            "bopp.extensions.importlib.metadata.entry_points",
            return_value=[ep1, ep2, valid_ep],
        ):
            REGISTRY.clear()
            # Warning is issued during update/initialization
            with pytest.warns(UserWarning, match="Conflict for extension 'conflict.schema'"):
                update_extensions()

            extensions = get_extensions()
            assert "conflict.schema" in extensions
            assert "valid.schema" in extensions

            # Accessing an unconflicted extension succeeds
            assert extensions["valid.schema"] is CustomItem

            # Accessing the conflicting extension raises BoppRegistryError
            with pytest.raises(
                BoppRegistryError,
                match="Conflicting extension registrations detected for schema 'conflict.schema'",
            ):
                _ = extensions["conflict.schema"]

            # Overriding explicitly resolves the conflict
            extensions["conflict.schema"] = CustomItem
            assert extensions["conflict.schema"] is CustomItem
    finally:
        reset_extensions()


def test_mark_conflict_without_prior_entry():
    """Verify mark_conflict initialization when the schema name had no prior entry (lines 94-97)."""
    reg = ExtensionRegistry()
    conflicting_entry_1 = "pkg_a:SchemaA"
    conflicting_entry_2 = "pkg_b:SchemaB"

    # First call covers lines 94-96: name not in _conflicts
    reg.mark_conflict("unregistered.conflict", conflicting_entry_1)
    assert reg._conflicts["unregistered.conflict"] == [conflicting_entry_1]

    # Second call covers branch 94 -> 97: name already in _conflicts
    reg.mark_conflict("unregistered.conflict", conflicting_entry_2)
    assert reg._conflicts["unregistered.conflict"] == [
        conflicting_entry_1,
        conflicting_entry_2,
    ]

    with pytest.raises(
        BoppRegistryError,
        match="Conflicting extension registrations detected for schema 'unregistered.conflict'",
    ):
        _ = reg["unregistered.conflict"]


def test_registry_getitem_key_error():
    """Verify KeyError when requesting an unregistered schema (line 114)."""
    reg = ExtensionRegistry()
    with pytest.raises(KeyError):
        _ = reg["missing.schema"]


def test_registry_import_module_spec_without_colon():
    """Verify module-only string import path and non-type resolution TypeError (lines 119, 128-130)."""
    reg = ExtensionRegistry()
    # Importing a module without ':' resolves the module object, which is not a type/union
    reg.register_entry("math_mod", "math")

    with pytest.raises(
        TypeError,
        match="Resolved extension for 'math_mod' must be a type, got module",
    ):
        _ = reg["math_mod"]


def test_registry_non_type_resolved_raises_type_error():
    """Verify TypeError when loaded entry point resolves to a non-type instance (lines 128-130)."""
    reg = ExtensionRegistry()
    reg.register_entry("not_a_type", 42)

    with pytest.raises(
        TypeError,
        match="Resolved extension for 'not_a_type' must be a type, got int",
    ):
        _ = reg["not_a_type"]


def test_registry_delitem_cleans_resolved_and_conflicts():
    """Verify __delitem__ purges entry from _raw_entries, _resolved, and _conflicts (line 136)."""
    reg = ExtensionRegistry()
    reg.register_entry("test.del", CustomItem)

    # Force resolution into _resolved
    _ = reg["test.del"]
    assert "test.del" in reg._resolved

    # Simulate a conflict record
    reg._conflicts["test.del"] = ["conflict"]

    del reg["test.del"]

    assert "test.del" not in reg._raw_entries
    assert "test.del" not in reg._resolved
    assert "test.del" not in reg._conflicts


def test_update_extensions_duplicate_identical_entry_point():
    """Verify duplicate entry points with identical values are ignored without warnings (lines 188-196)."""
    ep1 = SimpleNamespace(name="identical.schema", value="pkg_a:SchemaA")
    ep2 = SimpleNamespace(name="identical.schema", value="pkg_a:SchemaA")

    try:
        reset_extensions()
        with patch(
            "bopp.extensions.importlib.metadata.entry_points",
            return_value=[ep1, ep2],
        ):
            REGISTRY.clear()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                update_extensions()

            assert "identical.schema" in REGISTRY
            assert "identical.schema" not in REGISTRY._conflicts
    finally:
        reset_extensions()


def test_get_extensions_when_already_initialized():
    """Verify get_extensions returns the registry without re-invoking update_extensions when initialized (line 219)."""
    try:
        reset_extensions()
        assert ext_mod._INITIALIZED is True

        # Call get_extensions() when _INITIALIZED is True, executing branch 217 -> 219
        ext = get_extensions()
        assert ext is REGISTRY
    finally:
        reset_extensions()


def test_resolve_extensions_none_or_unset_payload():
    """Verify resolve_extensions safely exits when payload is None or UNSET."""
    for payload_val in (None, msgspec.UNSET):
        dummy_ann = SimpleNamespace(payload=payload_val)
        resolve_extensions(dummy_ann)
        assert dummy_ann.payload is payload_val


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


def test_resolved_extension_csv_roundtrip(tmp_path):
    """Verify that an annotation with already-resolved extension structs roundtrips cleanly to CSV."""
    exts = get_extensions()
    exts["org.test.custom"] = CustomItem

    try:
        ann = create(
            bopp_version="1.0",
            media_id="track:csv_resolved",
            payload_kind="ext",
            ext_schema="org.test.custom",
            value=[{"name": "snare", "score": 95.0}, {"name": "kick", "score": 100.0}],
            resolve_ext=True,
        )
        assert isinstance(ann.payload.value[0], CustomItem)

        csv_path = tmp_path / "resolved.csv"
        save_bopp_csv(ann, csv_path)

        reloaded = load_bopp_csv(csv_path, resolve_ext=True)
        assert isinstance(reloaded.payload.value[0], CustomItem)
        assert reloaded.payload.value[0].name == "snare"
        assert reloaded.payload.value[0].score == 95.0
        assert compute_annotation_id(reloaded) == ann.id
    finally:
        del exts["org.test.custom"]
