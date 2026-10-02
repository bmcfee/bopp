# Repository Design & Code Generation

This document provides an overview of the architecture and design of the `bopp` repository, including its code generation processes and guidance on how to make schema updates.

---

## Architecture Overview

`bopp` uses `msgspec` to provide fast, strongly-typed Python data structures for MIR (Music Information Retrieval) annotations, backed by JSON Schemas for validation and inter-language support.

Key modules and their roles:

* **`bopp.models`**: Contains automatically generated `msgspec.Struct` classes defining the models (Annotation root, Payloads, Extents, Confidences, and Metadata).
* **`bopp.registries`**: Auto-generated registry files (`registry_v1.py`, etc.) mapping string tags/identifiers to Python class types for dynamic parsing and validation.
* **`bopp.core`**: High-level factory (`create`) and validation functions.
* **`bopp.io`**: I/O routines for JSON, MsgPack, and CSV serialization/deserialization.
* **`bopp.transforms`**: Functional transformations on annotations (e.g., `trim`).
* **`bopp.util`**: Conversion utilities (DataFrame integration, header extraction).

---

## Automated Tooling & Workflows

Several code-generation and maintenance tasks are automated in this repository:

1. **Registry Generation**: Registries mapping string key tags to `msgspec.Struct` models are auto-generated to allow dynamic decoding across schema versions.
2. **JSON Schema Export**: Python `msgspec.Struct` definitions serve as the source of truth, and JSON Schemas are generated from them.
3. **Schema Documentation**: Documentation for schemas is compiled automatically from the generated JSON Schema files using `jsonschema2md`.

---

## Updating Schemas and Running Code Generation

To update existing schema definitions or add new payload/extent types:

### 1. Modify or Add Models
Edit or add model definitions under `bopp/models/v1/` (or the relevant version folder). Ensure your new structs inherit from `msgspec.Struct` and define the appropriate `tag` or `tag_field` if applicable.

### 2. Run Code Generation Scripts
We use [Hatch](https://hatch.pypa.io/) environments to automate generation tasks.

* **Generate Registries**:
  Re-generate `bopp/registries/registry_v<VERSION>.py` from the python models:
  ```bash
  hatch run generate-registry
  ```

* **Generate JSON Schemas**:
  Export JSON Schemas into the `schemas/` directory:
  ```bash
  hatch run generate-schemas
  ```

* **Generate Schema Documentation**:
  Re-generate Markdown schema docs from JSON Schemas:
  ```bash
  hatch run generate-schema-docs
  ```

* **Run All Generation Tasks**:
  To execute all generation steps sequentially:
  ```bash
  hatch run generate-all
  ```

---

## Development Workflow

When making schema changes, follow this general workflow:

1. Update Python models in `bopp/models/`.
2. Run `hatch run generate-all`.
3. Check for modified files (`git status` / `git diff`) to review changes to `bopp/registries/`, `schemas/`, and `docs/schema.md`.
4. Run tests to ensure compliance:
   ```bash
   hatch run test
   ```
