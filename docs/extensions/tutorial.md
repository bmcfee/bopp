# Tutorial: Drum Transcription Extension

This tutorial demonstrates how to create and distribute a drum transcription schema that restricts drum components to a closed vocabulary and records strike velocities.

You can also download this code as a packaged starter template:
{download}`bopp-extension-template.zip <../_static/bopp-extension-template.zip>`.

---

## 1. Define the Observation Struct (`my_drums/models.py`)

Create a `msgspec.Struct` that defines the schema for an individual observation item. Standard Python `enum.StrEnum` classes work well for closed-vocabulary labels:

```python
from enum import StrEnum
import msgspec

class DrumComponent(StrEnum):
    KICK = "kick"
    SNARE = "snare"
    HIHAT_CLOSED = "hihat_closed"
    HIHAT_OPEN = "hihat_open"
    TOM_LOW = "tom_low"
    TOM_HIGH = "tom_high"
    CRASH = "crash"
    RIDE = "ride"

class DrumStroke(msgspec.Struct, forbid_unknown_fields=True):
    """Observation item for a single drum strike."""
    component: DrumComponent
    velocity: int = 100  # MIDI velocity range [1, 127]
```

---

## 2. Register the Entry Point (`pyproject.toml`)

In your package's `pyproject.toml`, register the struct under the `"bopp_extension"` entry point group using a unique schema name:

```toml
[project.entry-points."bopp_extension"]
"org.example.drums:v1" = "my_drums.models:DrumStroke"
```

Install the package in editable mode:

```bash
pip install -e .
```

`bopp` will automatically discover the extension on demand.

---

## 3. Creating and Validating Extended Annotations

When calling `bopp.create()`, setting `resolve_ext=True` (the default) automatically validates and coerces the raw input dictionaries into instances of your registered struct:

```python
import bopp
from my_drums.models import DrumComponent, DrumStroke

ann = bopp.create(
    media_id="audio:funk_groove_01",
    payload_kind="ext",
    ext_schema="org.example.drums:v1",
    value=[
        {"component": "kick", "velocity": 110},
        {"component": "hihat_closed", "velocity": 85},
        {"component": "snare", "velocity": 115},
    ],
    extent_kind="time",
    time=[0.0, 0.25, 0.5],
)

# Values are strictly typed DrumStroke structs:
assert isinstance(ann.payload.value[0], DrumStroke)
assert ann.payload.value[0].component is DrumComponent.KICK
```

---

## 4. Serialization and Deserialization

Serialized files store data cleanly in standard JSON, MsgPack, or CSV formats:

```python
from bopp.io import load_bopp_json, save_bopp_json

save_bopp_json(ann, "groove.json")

# On load, resolve_ext=True automatically converts items back to DrumStroke
loaded = load_bopp_json("groove.json", resolve_ext=True)
assert isinstance(loaded.payload.value[0], DrumStroke)
```
