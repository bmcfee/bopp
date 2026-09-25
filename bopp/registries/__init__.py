from typing import Any

from .._version import get_registry_version
from ..exceptions import BoppRegistryError


def get_registry(version: str) -> dict[str, Any]:
    """Resolve the target registry and Annotation class based on the bopp_version.

    Parameters
    ----------
    version : str
        The version string identifying the BOPP version (e.g., "1.0.0", "v1", "1").

    Returns
    -------
    dict of str to Any
        A dictionary containing registry mappings and the target `Annotation` class:
        - "PAYLOAD_TYPE_REGISTRY": dict mapping payload tags to payload classes.
        - "EXTENT_TYPE_REGISTRY": dict mapping extent tags to extent classes.
        - "CONFIDENCE_TYPE_REGISTRY": dict mapping confidence tags to confidence classes.
        - "Annotation": the version-specific Annotation model class.

    Raises
    ------
    BoppRegistryError
        If the provided `version` is not supported.
    """
    registry_key = get_registry_version(version)

    if registry_key == "v1":
        from bopp.models.v1.annotation import Annotation

        from . import v1

        return {
            "PAYLOAD_TYPE_REGISTRY": v1.PAYLOAD_TYPE_REGISTRY,
            "EXTENT_TYPE_REGISTRY": v1.EXTENT_TYPE_REGISTRY,
            "CONFIDENCE_TYPE_REGISTRY": v1.CONFIDENCE_TYPE_REGISTRY,
            "Annotation": Annotation,
        }

    raise BoppRegistryError(f"Unsupported BOPP version: {version}")
