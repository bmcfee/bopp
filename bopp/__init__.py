"""BOPP: Bounded Observation Payload Protocol."""
from . import evaluation as evaluation
from . import extensions as extensions
from . import io as io
from . import models as models
from . import registries as registries
from . import transforms as transforms
from . import util as util
from ._version import (
    DEFAULT_SCHEMA_VERSION as DEFAULT_SCHEMA_VERSION,
)
from ._version import (
    __version__ as __version__,
)
from ._version import (
    get_current_schema_version as get_current_schema_version,
)
from ._version import (
    get_registry_version as get_registry_version,
)
from .core import (
    BoppArgumentError as BoppArgumentError,
)
from .core import (
    BoppArrayError as BoppArrayError,
)
from .core import (
    BoppError as BoppError,
)
from .core import (
    BoppIOError as BoppIOError,
)
from .core import (
    BoppRegistryError as BoppRegistryError,
)
from .core import (
    BoppValidationError as BoppValidationError,
)
from .core import (
    compute_annotation_id as compute_annotation_id,
)
from .core import (
    create as create,
)
from .core import (
    validate as validate,
)
from .core import (
    validate_and_set_annotation_id as validate_and_set_annotation_id,
)
from .evaluation import (
    evaluate as evaluate,
)
from .transforms import (
    filter_by as filter_by,
)
from .transforms import (
    to_times as to_times,
)
from .transforms import (
    trim as trim,
)

# Update the extensions registry
extensions.update_extensions()
