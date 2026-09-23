from typing import Any


def get_registry(version: str) -> dict[str, Any]:
    """
    Resolve the target registry and Annotation class based on the bopp_version.
    """
    if version in ("1.0.0", "v1", "1"):
        from bopp.models.v1.annotation import Annotation

        from . import v1
        
        return {
            "PAYLOAD_TYPE_REGISTRY": v1.PAYLOAD_TYPE_REGISTRY,
            "EXTENT_TYPE_REGISTRY": v1.EXTENT_TYPE_REGISTRY,
            "CONFIDENCE_TYPE_REGISTRY": v1.CONFIDENCE_TYPE_REGISTRY,
            "Annotation": Annotation
        }
        
    raise ValueError(f"Unsupported BOPP version: {version}")
