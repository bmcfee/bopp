        if isinstance(entry, type):
            resolved_type = entry
        elif hasattr(entry, "value") and ":" in str(entry.value):
            module_name, qualname = entry.value.split(":", 1)
            try:
                resolved_type = _lazy_loader.load(f"{module_name}:{qualname}")
            except (ModuleNotFoundError, ImportError):
                if hasattr(entry, "load"):
                    resolved_type = entry.load()
                else:
                    raise
        elif hasattr(entry, "load"):
            resolved_type = entry.load()
        else:
            resolved_type = entry
