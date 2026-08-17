import msgspec
import pyarrow as pa


class BoppBase(msgspec.Struct):
    """
    A base class for BOPP models. Dynamically validates parallel columnar 
    arrays on the root Annotation node, safely ignoring sub-models.
    """
    def __post_init__(self):
        # GUARD: Only run this on top-level Annotation containers
        if not hasattr(self, "payload"):
            return

        all_lengths = []

        extent_field = getattr(self, "extent", msgspec.UNSET)
        if extent_field is not msgspec.UNSET and extent_field is not None:
            extent_lengths = self._get_all_column_lengths(extent_field, "Extent")
            all_lengths.extend(extent_lengths)

        payload_field = getattr(self, "payload", msgspec.UNSET)
        if payload_field is not msgspec.UNSET and payload_field is not None:
            payload_lengths = self._get_all_column_lengths(payload_field, "Payload")
            all_lengths.extend(payload_lengths)
        
        confidence_field = getattr(self, "confidence", msgspec.UNSET)
        if confidence_field is not msgspec.UNSET and confidence_field is not None:
            conf_lengths = self._get_all_column_lengths(confidence_field, "Confidence")
            all_lengths.extend(conf_lengths)

        # Ensure all found arrays have the same length
        if len(set(all_lengths)) > 1:
            raise ValueError(
                f"Length mismatch: Found multiple array lengths {set(all_lengths)} "
                "across Extent, Payload, and Confidence facets."
            )

    @staticmethod
    def _get_all_column_lengths(facet_struct: msgspec.Struct, facet_name: str) -> list[int]:
        """
        Dynamically scans a msgspec struct for all list/array fields 
        and returns a list of their lengths.
        """
        lengths = []
        for field in msgspec.structs.fields(type(facet_struct)):
            val = getattr(facet_struct, field.name)
            
            if isinstance(val, (pa.Array, list)):
                lengths.append(len(val))
        
        if not lengths:
            raise ValueError(f"{facet_name} struct ({type(facet_struct).__name__}) contains no lists to measure.")
            
        return lengths
