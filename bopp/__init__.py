"""BOPP: Bounded Observation Payload Protocol."""
from . import io as io
from . import models as models
from . import registries as registries
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
    create as create,
)
from .core import (
    validate as validate,
)
