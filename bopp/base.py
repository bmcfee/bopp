import msgspec


class BoppBase(msgspec.Struct):
    """
    A base class for BOPP models. Dynamically validates parallel columnar 
    arrays on the root Annotation node, safely ignoring sub-models.
    """
    def __post_init__(self):
        # GUARD: Only run this on the top-level container
        if not (hasattr(self, "extent") and hasattr(self, "payload")):
            return

        # Dynamically measure the length of each facet
        extent_len = self._get_column_length(self.extent, "Extent")
        payload_len = self._get_column_length(self.payload, "Payload")
        
        print(f"Extent length: {extent_len}, Payload length: {payload_len}")
        if payload_len != extent_len:
            raise ValueError(
                f"Length mismatch: Extent contains {extent_len} items, "
                f"but Payload contains {payload_len} items."
            )
            
        confidence_field = getattr(self, "confidence", msgspec.UNSET)
        if confidence_field is not msgspec.UNSET and confidence_field is not None:
            conf_len = self._get_column_length(confidence_field, "Confidence")
            if conf_len != extent_len:
                raise ValueError(
                    f"Length mismatch: Extent contains {extent_len} items, "
                    f"but Confidence contains {conf_len} items."
                )

    @staticmethod
    def _get_column_length(facet_struct: msgspec.Struct, facet_name: str) -> int:
        """
        Dynamically scans a msgspec struct for its first list/array field 
        and returns its length, completely ignoring the field's name.
        """
        # msgspec.structs.fields() returns a tuple of field definitions for the struct
        for field in msgspec.structs.fields(type(facet_struct)):
            val = getattr(facet_struct, field.name)
            
            # Find the first field that is a list and return its length
            if isinstance(val, list):
                return len(val)
                
        # Fallback if no array is found (catches malformed schemas)
        raise ValueError(f"{facet_name} struct ({type(facet_struct).__name__}) contains no lists to measure.")
