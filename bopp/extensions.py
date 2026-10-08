from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Iterator, MutableMapping
from typing import Any

import lazy_loader as _lazy_loader

__all__ = ["ExtensionRegistry", "get_extensions", "reset_extensions", "update_extensions"]


class ExtensionRegistry(MutableMapping[str, type]):
    """
    Registry for BOPP extension payload schemas with lazy on-demand loading.

    Stores entry point representations and resolves the underlying schema types
    only when accessed.
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
        elif hasattr(entry, "value") and ":" in str(entry.value):
            module_name, qualname = entry.value.split(":", 1)
            try:
                resolved_type = _lazy_loader.load(f"{module_name}:{qualname}")
            except (ModuleNotFoundError, ImportError):
                if hasattr(entry, "load"):
                    resolved_type = entry.load()
                else:
                    raise
        elif hasattr(entry, "load"):
            resolved_type = entry.load()
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
