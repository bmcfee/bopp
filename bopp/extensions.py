import importlib.metadata

__all__ = ["get_extensions", "update_extensions", "reset_extensions"]

REGISTRY: dict[str, type] = {}

def update_extensions():
    # Query the specific entry point group
    eps = importlib.metadata.entry_points(group="bopp-extension")
    for ep in eps:
        if ep.name in REGISTRY:
            # TODO Handle or log conflicts if multiple packages register the same URI
            continue
        # TODO: make this a lazy load
        REGISTRY[ep.name] = ep.load()


def reset_extensions():
    REGISTRY.clear()
    update_extensions()


def get_extensions():
    return REGISTRY
