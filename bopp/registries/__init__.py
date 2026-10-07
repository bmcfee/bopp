    registry_key = get_registry_version(version)

    if registry_key == "v1":
        from . import v1

        return {
            "PAYLOAD_TYPE_REGISTRY": v1.PAYLOAD_TYPE_REGISTRY,
            "EXTENT_TYPE_REGISTRY": v1.EXTENT_TYPE_REGISTRY,
            "CONFIDENCE_TYPE_REGISTRY": v1.CONFIDENCE_TYPE_REGISTRY,
            "COMPLEX_FIELDS_REGISTRY": getattr(v1, "COMPLEX_FIELDS_REGISTRY", {}),
            "Annotation": v1.ANNOTATION_CLASS,
        }
