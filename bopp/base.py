from __future__ import annotations

import base64
import warnings
from typing import Any

import msgspec

from .exceptions import BoppArgumentError, BoppArrayError, BoppValidationError


class BoppBase(msgspec.Struct):
    """Base class for BOPP models.

    Dynamically validates parallel columnar arrays on the root Annotation node,
    safely ignoring sub-models.
    """
    def __post_init__(self):
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

        groupings_field = getattr(self, "groupings", msgspec.UNSET)
        if groupings_field is not msgspec.UNSET and groupings_field is not None:
            seen_names: set[str] = set()
            for g in groupings_field:
                g_name = getattr(g, "name", None)
                if isinstance(g_name, str):
                    if g_name in seen_names:
                        raise BoppValidationError(
                            f"Annotation contains multiple groupings with the same name: '{g_name}'"
                        )
                    seen_names.add(g_name)
                ids = getattr(g, "ids", None)
                if isinstance(ids, list):
                    all_lengths.append(len(ids))

        # Ensure all found arrays have the same length
        if len(set(all_lengths)) > 1:
            raise BoppArrayError(
                f"Length mismatch: Found multiple array lengths {set(all_lengths)} "
                "across Extent, Payload, Confidence, and Groupings facets."
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
            
            if isinstance(val, (list,)):
                lengths.append(len(val))
        
        if not lengths:
            raise BoppArgumentError(
                f"{facet_name} struct ({type(facet_struct).__name__}) contains no lists to measure."
            )
            
        return lengths

    def _repr_mimebundle_(
        self, include: list | None = None, exclude: list | None = None
    ) -> dict[str, Any]:
        """Return a MIME bundle dictionary for rich Jupyter display.

        Provides serialization suitable for front-end rendering extensions
        such as ``bopp-viewer``. The model is serialized to MessagePack format
        and encoded as an ASCII Base64 string under the MIME type
        ``application/vnd.bopp+msgpack``.

        Parameters
        ----------
        include : list or None, default=None
            MIME types to include. Maintained for IPython display protocol
            compatibility.
        exclude : list or None, default=None
            MIME types to exclude. Maintained for IPython display protocol
            compatibility.

        Returns
        -------
        dict of str to Any
            A MIME bundle dictionary mapping MIME type strings to their
            serialized representations.
        """
        bundle: dict[str, Any] = {}

        bundle["application/vnd.bopp+msgpack"] = base64.b64encode(msgspec.msgpack.encode(self)).decode("ascii")

        return bundle


class GroupingBase(msgspec.Struct):
    """Base class for BOPP Grouping objects.

    This primarily exists to implement a post-init check on distinctness of values
    if a label mapping is provided for group identifiers, and warn if non-negative
    group IDs are unmapped.
    """
    def __post_init__(self):
        labels = getattr(self, "labels", msgspec.UNSET)
        if labels is msgspec.UNSET or labels is None:
            return

        # Check that the labels mapping has distinct values
        if len(set(labels.values())) != len(labels):
            raise BoppValidationError("Grouping assigns two or more labels to the same group identifier")

        ids = getattr(self, "ids", msgspec.UNSET)
        if ids is not msgspec.UNSET and ids is not None:
            mapped_ids = set(labels.values())
            unmapped_non_negative = {i for i in ids if i >= 0 and i not in mapped_ids}
            if unmapped_non_negative:
                warnings.warn(
                    f"Grouping '{getattr(self, 'name', '')}' contains non-negative group IDs "
                    f"not included in label mapping: {sorted(unmapped_non_negative)}",
                    UserWarning,
                    stacklevel=2,
                )
