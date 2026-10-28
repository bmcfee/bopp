from __future__ import annotations

import base64
import warnings
from typing import Any

import msgspec

from .exceptions import BoppArrayError, BoppValidationError


class BoppBase(msgspec.Struct):
    """Base class for BOPP models.

    Dynamically validates parallel columnar arrays on the root Annotation node,
    safely ignoring sub-models.
    """

    def __post_init__(self) -> None:
        all_lengths: list[int] = []

        for attr_name, label in (
            ("extent", "Extent"),
            ("payload", "Payload"),
            ("confidence", "Confidence"),
        ):
            facet_field = getattr(self, attr_name, msgspec.UNSET)
            if facet_field is not msgspec.UNSET and facet_field is not None:
                all_lengths.extend(self._get_all_column_lengths(facet_field, label))

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

        extent_field = getattr(self, "extent", msgspec.UNSET)
        if extent_field is not msgspec.UNSET and extent_field is not None:
            from .util import _get_tag, is_disjoint, is_exhaustive

            tag = _get_tag(extent_field)
            # Special check for time_uniform: disjoint must be True
            if (
                tag == "time_uniform"
                and getattr(extent_field, "disjoint", None) is not True
            ):
                raise BoppValidationError("Uniform time extent requires disjoint=True.")

            if getattr(extent_field, "disjoint", False) is True and not is_disjoint(
                extent_field
            ):
                raise BoppValidationError(
                    f"Extent '{tag}' is marked disjoint=True, but observations overlap."
                )

            if getattr(extent_field, "exhaustive", False) is True:
                n_obs = all_lengths[0] if all_lengths else 0
                if not is_exhaustive(extent_field, N=n_obs):
                    raise BoppValidationError(
                        f"Extent '{tag}' is marked exhaustive=True, "
                        "but observations do not cover the specified bounds without gaps."
                    )

            if groupings_field is not msgspec.UNSET and groupings_field is not None:
                for g in groupings_field:
                    if getattr(g, "disjoint", False) is True and not is_disjoint(
                        extent_field, grouping=g
                    ):
                        raise BoppValidationError(
                            f"Grouping '{getattr(g, 'name', '')}' is marked disjoint=True, "
                            "but observations within at least one group overlap."
                        )
                    if getattr(g, "exhaustive", False) is True:
                        n_obs = all_lengths[0] if all_lengths else 0
                        if not is_exhaustive(extent_field, grouping=g, N=n_obs):
                            raise BoppValidationError(
                                f"Grouping '{getattr(g, 'name', '')}' is marked exhaustive=True, "
                                "but observations within at least one group do not cover bounds without gaps."
                            )
        else:
            if groupings_field is not msgspec.UNSET and groupings_field is not None:
                for g in groupings_field:
                    if getattr(g, "disjoint", False) is True:
                        raise BoppValidationError(
                            f"Grouping '{getattr(g, 'name', '')}' is marked disjoint=True, "
                            "but annotation has no extent."
                        )
                    if getattr(g, "exhaustive", False) is True:
                        raise BoppValidationError(
                            f"Grouping '{getattr(g, 'name', '')}' is marked exhaustive=True, "
                            "but annotation has no extent."
                        )

    @staticmethod
    def _get_all_column_lengths(
        facet_struct: msgspec.Struct | None, facet_name: str
    ) -> list[int]:
        """
        Dynamically scans a msgspec struct for all list/array fields
        and returns a list of their lengths.
        """
        if facet_struct is None or facet_struct is msgspec.UNSET:
            return []
        lengths = []
        for field in msgspec.structs.fields(type(facet_struct)):
            val = getattr(facet_struct, field.name)

            if isinstance(val, list):
                lengths.append(len(val))

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

        bundle["application/vnd.bopp+msgpack"] = base64.b64encode(
            msgspec.msgpack.encode(self)
        ).decode("ascii")

        return bundle


class GroupingBase(msgspec.Struct):
    """Base class for BOPP Grouping objects.

    This primarily exists to implement a post-init check on distinctness of values
    if a label mapping is provided for group identifiers, and warn if non-negative
    group IDs are unmapped.
    """

    def __post_init__(self) -> None:
        labels = getattr(self, "labels", msgspec.UNSET)
        if labels is msgspec.UNSET or labels is None:
            return

        # Check that the labels mapping has distinct values
        if len(set(labels.values())) != len(labels):
            raise BoppValidationError(
                "Grouping assigns two or more labels to the same group identifier"
            )

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
