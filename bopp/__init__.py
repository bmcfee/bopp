"""BOPP: Bounded Observation Payload Protocol."""

from . import (
    evaluation,
    extensions,
    io,
    models,
    registries,
    transforms,
    util,
)
from ._version import (
    DEFAULT_SCHEMA_VERSION,
    __version__,
    get_current_schema_version,
    get_registry_version,
)
from .core import (
    compute_annotation_id,
    create,
    ensure_annotation_id,
    validate,
    validate_annotation_id,
)
from .evaluation import evaluate
from .exceptions import (
    BoppArgumentError,
    BoppArrayError,
    BoppError,
    BoppIOError,
    BoppRegistryError,
    BoppValidationError,
)
from .extensions import resolve_extensions
from .transforms import (
    filter_by,
    to_score_interval,
    to_score_interval_float,
    to_score_quarter,
    to_score_quarter_float,
    to_times,
    trim,
)
from .util import (
    get_extent_bounds,
    infer_extent_bounds,
    is_disjoint,
    is_exhaustive,
)

__all__ = [
    "DEFAULT_SCHEMA_VERSION",
    "BoppArgumentError",
    "BoppArrayError",
    "BoppError",
    "BoppIOError",
    "BoppRegistryError",
    "BoppValidationError",
    "__version__",
    "compute_annotation_id",
    "create",
    "ensure_annotation_id",
    "evaluate",
    "evaluation",
    "extensions",
    "filter_by",
    "get_current_schema_version",
    "get_extent_bounds",
    "get_registry_version",
    "infer_extent_bounds",
    "io",
    "is_disjoint",
    "is_exhaustive",
    "models",
    "registries",
    "resolve_extensions",
    "to_score_interval",
    "to_score_interval_float",
    "to_score_quarter",
    "to_score_quarter_float",
    "to_times",
    "transforms",
    "trim",
    "util",
    "validate",
    "validate_annotation_id",
]
