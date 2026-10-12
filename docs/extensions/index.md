# Schema Extensions

`bopp` supports third-party domain extensions without requiring changes to the core JSON schemas. This mechanism allows domain-specific tasks—such as drum stroke transcription, expressive articulation nuances, or game-audio events—to benefit from strict `msgspec` validation, deterministic IDs, and seamless I/O serialization.

```{toctree}
:maxdepth: 2

model
guidelines
tutorial
```

## Starter Template

To jumpstart developing your own extension package, download the starter template:

* {download}`bopp-extension-template.zip <../_static/bopp-extension-template.zip>`

The template provides a minimal flit-based Python package containing a custom `msgspec.Struct` observation definition, `pyproject.toml` entry points, and editable install setup.
