from __future__ import annotations

import importlib
import importlib.metadata
import warnings
from collections.abc import Iterator, MutableMapping
from typing import Any

__all__ = ["ExtensionRegistry", "get_extensions", "reset_extensions", "update_extensions"]


class ExtensionRegistry(MutableMapping[str, type]):
    """
    Registry for BOPP extension payload schemas with lazy on-demand loading.

    Stores entry point representations and resolves the underlying schema types
    only when accessed.

    Notes
    -----
    Target Type Guidelines & Recommendations:

    Target types registered for extension payloads should be either standard Python
    built-in types (e.g., primitives, dictionaries, tuples) or `msgspec.Struct` subclasses.
    Using `msgspec.Struct` is strongly recommended for several key reasons:

    1. Serialization Equivalence:
       Annotation IDs are deterministic UUIDv5 hashes computed from canonical byte
       serialization. Both validation and serialization pipelines convert instances
       back and forth between target types and built-in primitives. Using `msgspec.Struct`
       ensures that the serialized representation is identical whether the extension
       schema is registered and resolved in the current environment, or remains as raw
       built-in dicts/primitives.

    2. Strict and Lossless Conversion:
       Extension conversion runs with `strict=True` to guarantee lossless conversions
       and avoid implicit coercion (such as string-to-float or int-to-float conversions)
       that would alter content hashes. `msgspec.Struct` provides native, fast, and
       strict validation during `msgspec.convert`.

    3. Unknown Fields Handling:
       Extension authors are recommended to configure structs with `forbid_unknown_fields=True`
       (e.g., `class MyObservation(msgspec.Struct, forbid_unknown_fields=True): ...`).
       This ensures unexpected fields trigger immediate validation errors rather than
       being silently discarded, which would corrupt the payload and produce hash mismatches.

    4. Default Values Considerations:
       Default values on extension struct fields should be used carefully. If an optional
       field has a default value on the struct, missing fields in serialized data will be
       populated upon conversion. Unless designed intentionally, this can introduce values
       that change the canonical re-serialized byte output. Authors should ensure payload
       data on the wire either fully specifies fields or that structs are configured to
       preserve exact wire semantics (e.g. using `omit_defaults=True`).
    """

    def __init__(self) -> None:
        self._raw_entries: dict[str, Any] = {}
        self._resolved: dict[str, type] = {}

    def register_entry(self, name: str, entry: Any) -> None:
        """
        Register an entry point or loader for an extension schema name.

        Parameters
        ----------
        name : str
            Extension schema identifier URI or name.
        entry : Any
            The importlib metadata entry point or lazy loader spec.
        """
        self._raw_entries[name] = entry
        self._resolved.pop(name, None)

    def __getitem__(self, key: str) -> type:
        if key in self._resolved:
            return self._resolved[key]

        if key not in self._raw_entries:
            raise KeyError(key)

        entry = self._raw_entries[key]
        if isinstance(entry, type):
            resolved_type = entry
        elif hasattr(entry, "load") and callable(entry.load):
            resolved_type = entry.load()
        elif isinstance(entry, str):
            if ":" in entry:
                mod_name, attr_name = entry.split(":", 1)
                mod = importlib.import_module(mod_name)
                resolved_type = getattr(mod, attr_name)
            else:
                resolved_type = importlib.import_module(entry)
        else:
            resolved_type = entry

        self._resolved[key] = resolved_type
        return resolved_type

    def __setitem__(self, key: str, value: type) -> None:
        self._raw_entries[key] = value
        self._resolved[key] = value

    def __delitem__(self, key: str) -> None:
        del self._raw_entries[key]
        self._resolved.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw_entries)

    def __len__(self) -> int:
        return len(self._raw_entries)

    def clear(self) -> None:
        """Clear all registered and resolved extension entries."""
        self._raw_entries.clear()
        self._resolved.clear()


REGISTRY = ExtensionRegistry()
_INITIALIZED: bool = False


def update_extensions() -> None:
    """
    Discover and register installed entry points under group 'bopp-extension'.

    Warns if multiple installed packages register conflicting extension schemas
    under the same identifier.
    """
    global _INITIALIZED
    eps = importlib.metadata.entry_points(group="bopp-extension")
    for ep in eps:
        if ep.name in REGISTRY:
            existing = REGISTRY._raw_entries[ep.name]
            existing_val = getattr(existing, "value", existing)
            new_val = getattr(ep, "value", ep)
            if existing_val != new_val:
                warnings.warn(
                    f"Conflict for extension '{ep.name}': already registered as "
                    f"'{existing_val}', ignoring conflicting registration '{new_val}'.",
                    UserWarning,
                    stacklevel=2,
                )
            continue
        REGISTRY.register_entry(ep.name, ep)
    _INITIALIZED = True


def reset_extensions() -> None:
    """Clear all registered extensions and re-run entry point discovery."""
    global _INITIALIZED
    REGISTRY.clear()
    _INITIALIZED = False
    update_extensions()


def get_extensions() -> ExtensionRegistry:
    """
    Retrieve the global extension registry, discovering extensions if needed.

    Returns
    -------
    ExtensionRegistry
        Mutable mapping of schema identifiers to extension types.
    """
    if not _INITIALIZED:
        update_extensions()
    return REGISTRY
