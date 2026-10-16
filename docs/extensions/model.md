# Extension Model & Registry

## The `"ext"` Payload

Extensions in `bopp` leverage the core `"ext"` payload type (`ExtensionPayload`). In the core schema, the `"ext"` payload specifies:

* **`payload_type`**: Always `"ext"`.
* **`ext_schema`**: A unique string URI identifying the target schema definition (e.g., `"org.example.drums:v1"`).
* **`value`**: An array of observation items. In the initial decode pass, items can be generic dictionaries or primitives, which are subsequently validated and converted against a registered schema type during resolution.

## Registration and Lazy Loading

Extensions register their types with Python's standard packaging entry point mechanism under the group name **`bopp_extension`**.

To prevent import overhead when loading annotations, `bopp.extensions` registers metadata entry points lazily. Module imports and type resolution are deferred until the specific `ext_schema` is encountered and accessed via `bopp.extensions.get_extensions()`.

### Handling Conflicts

If multiple installed packages register conflicting definitions for the same `ext_schema` identifier, `bopp` emits a warning (`UserWarning`) during entry-point discovery and raises `BoppRegistryError` when that schema is accessed until the conflict is explicitly overridden in `get_extensions()`.

## Missing Extensions and Strictness

When loading an annotation that uses an `ext_schema` not installed in the local environment:

* **Default behavior (`allow_missing=True`)**: `bopp.io.resolve_extensions()` issues a `UserWarning` and leaves `ann.payload.value` as generic dictionaries or primitives. This allows environments without the extension package to still inspect and process other facets of the annotation.
* **Strict behavior (`allow_missing=False`)**: Raises `bopp.exceptions.BoppRegistryError` if the required extension cannot be resolved.
