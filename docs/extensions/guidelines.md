# Target Types & Validation Contracts

## Target Types: Structs and Unions

Extension schemas can map to standard built-in types (e.g. `dict`, `int`, `str`), `msgspec.Struct` classes, or tagged union types (`A | B` or `typing.Union[A, B]`).

Authors are strongly recommended to use `msgspec.Struct` as the base class for custom observation items:

1. **Serialization Equivalence and ID Stability**:
   `msgspec` encodes `Struct` instances to JSON and MsgPack with byte-level parity to plain dictionaries. An annotation produces the exact same deterministic content hash and UUID whether it is evaluated in an environment with the extension package installed (as typed structs) or in an environment where it remains as raw built-in dictionaries.

2. **Native `msgspec` Integration**:
   Using `msgspec.Struct` keeps extension validation and serialization within `msgspec` and avoids third-party validator dependencies.

3. **Lossless Conversion with Strict Validation**:
   When resolving extension payloads with `msgspec.convert(..., strict=True)`, `msgspec.Struct` models enforce exact type compliance without silent loss of precision or unexpected type casting.

4. **Guarding Against Field Stripping (`forbid_unknown_fields=True`)**:
   Standard structs ignore unrecognized fields by default. If unrecognized fields are dropped during conversion to a struct, the re-serialized payload loses data, causing validation and hash mismatches. Extension authors should set `forbid_unknown_fields=True` on their structs to ensure invalid or undeclared fields fail loudly:
   ```python
   class DrumStroke(msgspec.Struct, forbid_unknown_fields=True):
       component: str
       velocity: int
   ```

5. **Precautions with Default Field Values**:
   If an extension struct defines default field values, deserializing payloads that omitted those fields on disk will explicitly instantiate the defaults. When serialized again, the presence of these new fields can alter the content hash unless the schema is configured with `omit_defaults=True` or authors ensure fully populated records.

6. **Tagged Unions**:
   When using a union of multiple structs (`TypeA | TypeB`), all structs in the union must be tagged (e.g. `tag=True` or `tag_field="..."`) so that `msgspec` can reliably disambiguate records during resolution.

## Best Practices

* **Namespace Collisions**: Use reverse-domain prefixes (e.g. `org.mir_tools.drums:v1`) to avoid schema name conflicts across different libraries.
* **Serialization Compatibility**: Rely on `msgspec`-supported types (`StrEnum`, primitives, typed `msgspec.Struct`) so items roundtrip cleanly to JSON, binary MsgPack, and CSV frontmatter without custom encoder hooks.
