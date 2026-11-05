from __future__ import annotations

import copy
import uuid
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

import msgspec

from ._version import __version__, get_current_schema_version
from .exceptions import (
    BoppArgumentError,
    BoppArrayError,
    BoppRegistryError,
    BoppValidationError,
)
from .extensions import resolve_extensions
from .registries import (
    get_grouping_class,
    get_metadata_class,
    get_registry,
)
from .util import (
    EXTENT_AXES_METADATA,
    FRACTION_EXTENT_TAGS,
    _column_item_type,
    _derive_parents_and_sandbox,
    _get_facet_list_fields,
    _get_tag,
    _to_coord,
    infer_extent_bounds,
)

if TYPE_CHECKING:
    from .base import BoppBase

BOPP_NAMESPACE = uuid.UUID("a9b3e1c0-42f8-4e89-8d62-b9123456789a")

__all__ = [
    "BOPP_NAMESPACE",
    "append",
    "compute_annotation_id",
    "create",
    "create_grouping",
    "create_metadata",
    "create_sandbox",
    "ensure_annotation_id",
    "extend",
    "resolve_extensions",
    "validate",
    "validate_annotation_id",
]


def compute_annotation_id(annotation: Any) -> str:
    """
    Compute a deterministic UUIDv5 identifier for an annotation.

    The UUID is calculated over the canonical JSON serialization of the
    annotation content, excluding any existing 'id' field.

    Parameters
    ----------
    annotation : Any
        An Annotation struct instance or dictionary representation of an annotation.

    Returns
    -------
    str
        The computed UUIDv5 string representation.
    """
    data = msgspec.to_builtins(annotation, order="deterministic")
    if "id" in data:
        del data["id"]
    canonical_bytes = msgspec.json.encode(data, order="deterministic")
    return str(uuid.uuid5(BOPP_NAMESPACE, canonical_bytes))


def validate_annotation_id(annotation: Any) -> bool:
    """
    Validate the UUIDv5 identifier of an annotation without silently mutating it.

    If an 'id' is present, validates it against the computed UUIDv5.

    Parameters
    ----------
    annotation : Any
        An Annotation struct instance to validate.

    Returns
    -------
    bool
        True if the ID is present and valid, False if ID is missing.

    Raises
    ------
    BoppValidationError
        If an existing 'id' on the annotation does not match the computed ID.
    """
    existing_id = getattr(annotation, "id", None)

    if existing_id is None or existing_id is msgspec.UNSET:
        return False

    computed_id = compute_annotation_id(annotation)

    if existing_id != computed_id:
        raise BoppValidationError(
            f"Annotation ID mismatch: object has '{existing_id}', computed '{computed_id}'"
        )

    return True


def ensure_annotation_id(annotation: Any) -> str:
    """
    Compute and set the UUIDv5 identifier for an annotation if missing, or validate it if present.

    Parameters
    ----------
    annotation : Any
        An Annotation struct instance.

    Returns
    -------
    str
        The assigned or validated UUIDv5 identifier.

    Raises
    ------
    BoppValidationError
        If an existing 'id' does not match the computed ID.
    """
    if not validate_annotation_id(annotation):
        annotation.id = compute_annotation_id(annotation)
    return str(annotation.id)


def _extract_kwargs(
    cls: type[msgspec.Struct], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """
    Extract and remove keys belonging to the target msgspec.Struct class.

    Parameters
    ----------
    cls : type of msgspec.Struct
        Target class whose valid field names will be extracted.
    kwargs : dict
        Dictionary of keyword arguments to extract matching keys from.
        This dictionary is mutated in place.

    Returns
    -------
    dict
        Dictionary of extracted keyword arguments relevant to `cls`.
    """
    valid_keys = cls.__struct_fields__
    return {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid_keys}


def create_metadata(
    metadata_type: str | dict[str, Any],
    *,
    bopp_version: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Create an AnnotationMetadata Struct instance via registry lookup.

    Parameters
    ----------
    metadata_type : str or dict of str to Any
        The metadata discriminator tag (e.g. 'human', 'algorithm', 'derived', 'crowd', 'sensor', 'other')
        or a dictionary containing metadata fields.
    bopp_version : str or None, optional
        Schema version string to select the registry. Defaults to current version.
    **kwargs : Any
        Keyword arguments matching the fields of the target metadata Struct.

    Returns
    -------
    msgspec.Struct
        An instantiated metadata struct.

    Raises
    ------
    BoppRegistryError
        If metadata_type is unrecognized.
    BoppArgumentError
        If metadata arguments are missing, invalid, or contradictory.
    """
    if isinstance(metadata_type, dict):
        meta_dict = dict(metadata_type)
        meta_dict.update(kwargs)
        tag = meta_dict.pop("metadata_type", None)
        if tag is None:
            raise BoppArgumentError(
                "Metadata dictionary must contain 'metadata_type' key."
            )
        cls = get_metadata_class(tag, version=bopp_version)
        try:
            return msgspec.convert(meta_dict, cls)
        except (msgspec.ValidationError, TypeError) as err:
            raise BoppValidationError(
                f"Validation failed for metadata '{tag}': {err}"
            ) from err

    cls = get_metadata_class(metadata_type, version=bopp_version)
    try:
        return cls(**kwargs)
    except TypeError as err:
        raise BoppArgumentError(
            f"Failed to instantiate metadata '{metadata_type}': {err}"
        ) from err


def create_grouping(
    name: str,
    ids: Sequence[int] | Sequence[str] | None = None,
    *,
    labels: dict[str, int] | Sequence[str] | None = None,
    disjoint: bool = False,
    exhaustive: bool = False,
    bopp_version: str | None = None,
) -> Any:
    """
    Create a Grouping Struct instance without importing from schema-versioned modules.

    Supports integer IDs with optional dictionary label mapping, string category
    labels (auto-enumerating unique IDs), or a sequence of labels passed via
    the ``labels`` parameter.

    Parameters
    ----------
    name : str
        Name of the grouping layer.
    ids : Sequence of int, Sequence of str, or None, optional
        Parallel array of group IDs or string category labels.
    labels : dict of str to int, Sequence of str, or None, optional
        Mapping from string labels to integer group IDs, or a sequence of string
        labels corresponding to each observation.
    disjoint : bool, default False
        Indicates whether observations within each group are non-overlapping.
    exhaustive : bool, default False
        Indicates whether observations within each group cover the extent without gaps.
    bopp_version : str or None, optional
        Schema version string to select the underlying Grouping class.

    Returns
    -------
    msgspec.Struct
        An instantiated Grouping struct.

    Raises
    ------
    BoppArgumentError
        If inputs are contradictory or invalid.
    BoppValidationError
        If Grouping validation fails.
    """
    Grouping_cls = get_grouping_class(version=bopp_version)

    # Case 1: labels is a sequence of strings and ids is None
    if (
        ids is None
        and isinstance(labels, Sequence)
        and not isinstance(labels, (dict, str, bytes))
    ):
        ids = labels
        labels = None

    # Case 2: labels is a sequence of strings and ids is provided
    if isinstance(labels, Sequence) and not isinstance(labels, (dict, str, bytes)):
        for lbl in labels:
            if not isinstance(lbl, str):
                raise BoppArgumentError(
                    f"Grouping labels sequence must contain strings; found {type(lbl).__name__} ({lbl!r})."
                )
        if len(set(labels)) != len(labels):
            raise BoppArgumentError(
                f"Duplicate labels found in grouping labels sequence: {list(labels)}."
            )
        labels = {lbl: idx for idx, lbl in enumerate(labels)}

    int_ids: list[int]
    labels_val: dict[str, int] | msgspec.UnsetType

    if ids is not None and len(ids) > 0 and all(isinstance(x, str) for x in ids):
        # String categorical labels in ids
        if isinstance(labels, dict):
            int_ids = []
            for s in ids:
                if s not in labels:
                    raise BoppArgumentError(
                        f"Label '{s}' in ids was not found in the provided labels mapping."
                    )
                int_ids.append(labels[s])
            labels_val = labels
        else:
            # Auto-assign IDs 0, 1, 2... in order of appearance
            unique_map: dict[str, int] = {}
            int_ids = []
            for s in ids:
                if s not in unique_map:
                    unique_map[s] = len(unique_map)
                int_ids.append(unique_map[s])
            labels_val = unique_map
    elif ids is not None:
        int_ids = []
        for x in ids:
            if not isinstance(x, int) or isinstance(x, bool):
                raise BoppArgumentError(
                    f"Grouping IDs must be integers or strings; found {type(x).__name__} ({x!r})."
                )
            int_ids.append(x)
        labels_val = labels if isinstance(labels, dict) else msgspec.UNSET
    else:
        int_ids = []
        labels_val = labels if isinstance(labels, dict) else msgspec.UNSET

    try:
        return Grouping_cls(
            name=name,
            ids=int_ids,
            disjoint=disjoint,
            exhaustive=exhaustive,
            labels=labels_val,
        )
    except (msgspec.ValidationError, TypeError) as err:
        raise BoppValidationError(f"Failed to create Grouping '{name}': {err}") from err


def create_sandbox(
    data: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Create and validate a sandbox dictionary for unstructured user-defined data.

    Parameters
    ----------
    data : dict of str to Any or None, optional
        Initial dictionary of sandbox key-value pairs.
    **kwargs : Any
        Additional key-value pairs to store in the sandbox.

    Returns
    -------
    dict of str to Any
        Validated sandbox dictionary.

    Raises
    ------
    BoppArgumentError
        If data is not a dictionary.
    BoppValidationError
        If values cannot be encoded to JSON.
    """
    sandbox_dict: dict[str, Any] = {}
    if data is not None:
        if not isinstance(data, dict):
            raise BoppArgumentError(
                f"sandbox data must be a dictionary, got {type(data).__name__}."
            )
        sandbox_dict.update(data)
    sandbox_dict.update(kwargs)

    try:
        msgspec.json.encode(sandbox_dict)
    except (msgspec.EncodeError, TypeError) as err:
        raise BoppValidationError(
            f"Sandbox data cannot be serialized to JSON: {err}"
        ) from err

    return sandbox_dict


def create(
    *,
    media_id: str,
    payload_type: str,
    extent_type: str | None = None,
    confidence_type: str | None = None,
    metadata_type: str | None = None,
    metadata: Any = msgspec.UNSET,
    groupings: list[Any] | Any = msgspec.UNSET,
    grouping: Any = None,
    parents: list[str] | Any = msgspec.UNSET,
    bopp_version: str | None = None,
    sandbox: Any = msgspec.UNSET,
    empty: bool = False,
    generate_id: bool = True,
    resolve_ext: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create a new Annotation instance for the specified BOPP version.

    Parameters
    ----------
    media_id : str
        Unique media identifier associated with the annotation.
    payload_type : str
        Type identifier registered for the payload struct.
    extent_type : str or None, optional
        Type identifier registered for the extent struct, if applicable.
    confidence_type : str or None, optional
        Type identifier registered for the confidence struct, if applicable.
    metadata_type : str or None, optional
        Type identifier registered for the metadata struct, if applicable.
    metadata : msgspec.Struct or dict, optional
        Explicit metadata struct instance or dictionary representation.
    groupings : list of Grouping, list of dict, Grouping, or dict, optional
        One or more Grouping struct instances or dictionary representations.
    grouping : Grouping or dict, optional
        Convenience alias for specifying a single grouping layer. Mutually exclusive
        with `groupings`.
    parents : list of str, optional
        List of parent annotation UUIDs from which this annotation was derived.
    bopp_version : str or None, optional
        Schema version string to select the underlying type registry.
        If None, defaults to the current default schema version.
    sandbox : Any, optional
        Unstructured storage area for arbitrary user-defined data.
    empty : bool, default False
        If True, unprovided columnar array fields on payload, extent, and confidence
        facets are automatically initialized to empty lists (`[]`).
    generate_id : bool, default True
        If True, computes and assigns deterministic UUIDv5 `id` to the annotation.
    resolve_ext : bool, default True
        If True and payload_type is 'ext', resolves and validates the extension payload values.
    **kwargs : Any
        Keyword arguments matching fields for the payload, extent, confidence,
        or metadata structures.

    Returns
    -------
    Annotation
        An instantiated Annotation structure.

    Raises
    ------
    BoppRegistryError
        If an unrecognized `payload_type`, `extent_type`, `confidence_type`, or
        `metadata_type` is provided.
    BoppArgumentError
        If unused keyword arguments remain, conflicting parameters are passed, or
        required fields are missing.
    """
    if bopp_version is None:
        bopp_version = get_current_schema_version()

    if grouping is not None:
        if groupings is not msgspec.UNSET and groupings is not None:
            raise BoppArgumentError("Cannot specify both 'grouping' and 'groupings'.")
        groupings = [grouping]
    elif groupings is not msgspec.UNSET and groupings is not None:
        if isinstance(groupings, (dict, msgspec.Struct)):
            groupings = [groupings]

    registry = get_registry(bopp_version)

    PAYLOAD_TYPE_REGISTRY = registry["PAYLOAD_TYPE_REGISTRY"]
    EXTENT_TYPE_REGISTRY = registry["EXTENT_TYPE_REGISTRY"]
    CONFIDENCE_TYPE_REGISTRY = registry["CONFIDENCE_TYPE_REGISTRY"]
    METADATA_TYPE_REGISTRY = registry.get("METADATA_TYPE_REGISTRY", {})
    Annotation = registry["Annotation"]
    Grouping = registry.get("Grouping")

    try:
        payload_cls = PAYLOAD_TYPE_REGISTRY[payload_type]
    except KeyError as e:
        raise BoppRegistryError(f"Unrecognized payload type: {e}") from e

    payload_args = _extract_kwargs(payload_cls, kwargs)
    if empty:
        for f in msgspec.structs.fields(payload_cls):
            if f.name not in payload_args and _column_item_type(f.type) is not None:
                payload_args[f.name] = []
    payload_obj = payload_cls(**payload_args)

    # Extract kwargs by mutating the dictionary
    extent_obj = msgspec.UNSET
    if extent_type is not None:
        try:
            extent_cls = EXTENT_TYPE_REGISTRY[extent_type]
        except KeyError as e:
            raise BoppRegistryError(f"Unrecognized extent type: {e}") from e

        extent_args = _extract_kwargs(extent_cls, kwargs)
        extent_args.setdefault("disjoint", extent_type == "time_uniform")
        extent_args.setdefault("exhaustive", False)
        if empty and extent_type != "time_uniform":
            for f in msgspec.structs.fields(extent_cls):
                if f.name not in extent_args and _column_item_type(f.type) is not None:
                    extent_args[f.name] = []
        extent_obj = extent_cls(**extent_args)

    confidence_obj = msgspec.UNSET
    if confidence_type:
        try:
            confidence_cls = CONFIDENCE_TYPE_REGISTRY[confidence_type]
        except KeyError as e:
            raise BoppRegistryError(f"Unrecognized confidence type: {e}") from e

        confidence_args = _extract_kwargs(confidence_cls, kwargs)
        if empty:
            for f in msgspec.structs.fields(confidence_cls):
                if (
                    f.name not in confidence_args
                    and _column_item_type(f.type) is not None
                ):
                    confidence_args[f.name] = []
        confidence_obj = confidence_cls(**confidence_args)

    metadata_obj = metadata
    if metadata_type is not None:
        try:
            metadata_cls = METADATA_TYPE_REGISTRY[metadata_type]
        except KeyError as e:
            raise BoppRegistryError(f"Unrecognized metadata type: {e}") from e

        if metadata is not msgspec.UNSET and metadata is not None:
            if isinstance(metadata, dict):
                meta_dict = dict(metadata)
                meta_tag = meta_dict.pop("metadata_type", None)
                if meta_tag is not None and meta_tag != metadata_type:
                    raise BoppArgumentError(
                        f"Conflicting metadata_type: '{metadata_type}' vs dict '{meta_tag}'."
                    )
                metadata_args = _extract_kwargs(metadata_cls, kwargs)
                meta_dict.update(metadata_args)
                try:
                    metadata_obj = metadata_cls(**meta_dict)
                except TypeError as err:
                    raise BoppArgumentError(
                        f"Failed to instantiate metadata '{metadata_type}': {err}"
                    ) from err
            else:
                raise BoppArgumentError(
                    "Cannot specify both 'metadata' and 'metadata_type'."
                )
        else:
            metadata_args = _extract_kwargs(metadata_cls, kwargs)
            metadata_obj = metadata_cls(**metadata_args)
    elif isinstance(metadata, dict):
        metadata_obj = create_metadata(metadata, bopp_version=bopp_version)

    groupings_obj = groupings
    if (
        groupings is not msgspec.UNSET
        and groupings is not None
        and Grouping is not None
    ):
        groupings_list = []
        for g in groupings:
            if isinstance(g, dict):
                groupings_list.append(create_grouping(**g, bopp_version=bopp_version))
            else:
                groupings_list.append(g)
        groupings_obj = groupings_list

    if (
        sandbox is not msgspec.UNSET
        and sandbox is not None
        and isinstance(sandbox, dict)
    ):
        sandbox = create_sandbox(sandbox)

    # Any remaining kwargs indicate a user typo or a schema mismatch
    if kwargs:
        raise BoppArgumentError(f"Unconsumed keyword arguments: {list(kwargs.keys())}")

    ann = Annotation(
        media_id=media_id,
        parents=parents,
        bopp_version=bopp_version,
        metadata=metadata_obj,
        extent=extent_obj,
        payload=payload_obj,
        confidence=confidence_obj,
        sandbox=sandbox,
        groupings=groupings_obj,
    )

    if resolve_ext and payload_type == "ext":
        resolve_extensions(ann)

    if generate_id:
        ensure_annotation_id(ann)

    return ann


def _get_annotation_columns(
    annotation: Any,
) -> tuple[
    dict[str, list[Any]],
    dict[str, list[Any]],
    dict[str, list[Any]],
    list[Any],
]:
    payload = getattr(annotation, "payload", None)
    payload_cols = _get_facet_list_fields(payload)

    extent = getattr(annotation, "extent", None)
    extent_tag = _get_tag(extent) if extent is not None else None
    if extent_tag == "time_uniform":
        extent_cols = {}
    else:
        extent_cols = _get_facet_list_fields(extent)

    confidence = getattr(annotation, "confidence", None)
    confidence_cols = _get_facet_list_fields(confidence)

    raw_groupings = getattr(annotation, "groupings", None)
    groupings_list = (
        [] if raw_groupings in (None, msgspec.UNSET) else list(raw_groupings)
    )

    return payload_cols, extent_cols, confidence_cols, groupings_list


def append(
    annotation: BoppBase,
    observation: dict[str, Any] | None = None,
    *,
    inplace: bool = False,
    groupings: dict[str, Any] | Sequence[Any] | None = None,
    update_bounds: bool = False,
    derive: bool = True,
    **kwargs: Any,
) -> BoppBase:
    """
    Append a single observation to an existing Annotation.

    Parameters
    ----------
    annotation : BoppBase
        The Annotation struct instance to append to.
    observation : dict of str to Any or None, optional
        Key-value mapping of observation values for payload, extent, and confidence facets.
    inplace : bool, default False
        If True, mutates the annotation's columnar lists in place and voids `annotation.id`
        (setting it to UNSET). Preserves original metadata and parents.
        If False, returns a new derived annotation with updated parents, derived metadata,
        and a recomputed UUIDv5 id.
    groupings : dict or sequence, optional
        Group assignments for each grouping layer in the annotation.
        Can be a dict mapping `{layer_name: group_id_or_label}` or a sequence of IDs
        matching the order of `annotation.groupings`.
    update_bounds : bool, default False
        If True, updates explicit min/max bounds on the extent if the new observation
        falls outside them. If False and bounds are exceeded, emits a UserWarning.
    derive : bool, default True
        Used when inplace=False. If True, sets metadata to DerivedAnnotationMetadata;
        if False, preserves a copy of the parent metadata.
    **kwargs : Any
        Observation fields passed as keyword arguments. Grouping layer names can also
        be passed directly as kwargs if they do not collide with facet column names.

    Returns
    -------
    BoppBase
        The mutated annotation (if inplace=True) or a new derived annotation (if inplace=False).

    Raises
    ------
    BoppArgumentError
        If required facet fields are missing, unconsumed fields are passed, or grouping
        values cannot be resolved.
    """
    target = annotation if inplace else copy.deepcopy(annotation)
    (
        payload_cols,
        extent_cols,
        confidence_cols,
        groupings_list,
    ) = _get_annotation_columns(target)

    all_cols = (
        list(payload_cols.values())
        + list(extent_cols.values())
        + list(confidence_cols.values())
        + [g.ids for g in groupings_list if getattr(g, "ids", None) is not None]
    )
    n_obs = len(all_cols[0]) if all_cols else 0

    obs = dict(observation) if observation is not None else {}
    obs.update(kwargs)

    group_assignments: dict[str, Any] = {}
    if groupings is not None:
        if isinstance(groupings, dict):
            registered_grouping_names = {g.name for g in groupings_list}
            unknown_groupings = set(groupings.keys()) - registered_grouping_names
            if unknown_groupings:
                raise BoppArgumentError(
                    f"Unknown grouping layer(s) for append: {sorted(unknown_groupings)}."
                )
            group_assignments.update(groupings)
        elif isinstance(groupings, Sequence) and not isinstance(
            groupings, (str, bytes)
        ):
            if len(groupings) != len(groupings_list):
                raise BoppArgumentError(
                    f"Expected {len(groupings_list)} grouping values, got {len(groupings)}."
                )
            for idx, g in enumerate(groupings_list):
                group_assignments[g.name] = groupings[idx]
        else:
            raise BoppArgumentError(
                "groupings must be a dictionary or sequence of group assignments."
            )

    for g in groupings_list:
        if g.name in obs and g.name not in group_assignments:
            group_assignments[g.name] = obs.pop(g.name)

    # Check required facet fields
    missing: list[str] = []
    for fname in payload_cols:
        if fname not in obs:
            missing.append(f"payload:{fname}")
    for fname in extent_cols:
        if fname not in obs:
            missing.append(f"extent:{fname}")
    for fname in confidence_cols:
        if fname not in obs:
            missing.append(f"confidence:{fname}")
    for g in groupings_list:
        if g.name not in group_assignments:
            missing.append(f"grouping:{g.name}")

    if missing:
        raise BoppArgumentError(f"Missing required field(s) for append: {missing}")

    valid_keys = (
        set(payload_cols.keys()) | set(extent_cols.keys()) | set(confidence_cols.keys())
    )
    extra = set(obs.keys()) - valid_keys
    if extra:
        raise BoppArgumentError(f"Unconsumed argument(s) for append: {sorted(extra)}")

    # Resolve grouping values to IDs
    resolved_gids: list[tuple[Any, int]] = []
    for g in groupings_list:
        val = group_assignments[g.name]
        labels = getattr(g, "labels", msgspec.UNSET)
        if isinstance(val, str):
            if labels not in (None, msgspec.UNSET) and isinstance(labels, dict):
                if val in labels:
                    gid = labels[val]
                else:
                    gid = max(labels.values(), default=-1) + 1
                    labels[val] = gid
            else:
                g.labels = {val: 0}
                gid = 0
        else:
            if not isinstance(val, int) or isinstance(val, bool):
                raise BoppArgumentError(
                    f"Invalid group ID for grouping '{g.name}': expected integer or string, got {type(val).__name__} ({val!r})."
                )
            gid = val
        resolved_gids.append((g, gid))

    # Cheap Sanity Checks
    extent = getattr(target, "extent", None)
    extent_tag = _get_tag(extent) if extent is not None else None
    if extent is not None and extent is not msgspec.UNSET:
        is_fraction = extent_tag in FRACTION_EXTENT_TAGS
        specs = EXTENT_AXES_METADATA.get(extent_tag, [])
        for spec in specs:
            dim = spec["dim"]
            min_attr = spec["min_attr"]
            max_attr = spec["max_attr"]
            kind = spec["kind"]
            if kind == "point":
                coord = _to_coord(obs[spec["field"]])
                obs_min, obs_max = coord, coord
            elif kind == "origin_span":
                orig = _to_coord(obs[spec["origin"]])
                span = _to_coord(obs[spec["span"]])
                obs_min, obs_max = orig, orig + span
            elif kind == "min_max":
                obs_min = _to_coord(obs[spec["min_field"]])
                obs_max = _to_coord(obs[spec["max_field"]])
            else:  # kind == "uniform"
                st = float(extent.start_time)
                sp = float(extent.sampling_period)
                t = st + n_obs * sp
                obs_min, obs_max = t, t

            curr_min = getattr(extent, min_attr, msgspec.UNSET)
            if curr_min not in (None, msgspec.UNSET):
                min_coord = _to_coord(curr_min)
                if obs_min < min_coord:
                    if update_bounds:
                        new_val = (
                            [obs_min.numerator, obs_min.denominator]
                            if is_fraction
                            else obs_min
                        )
                        setattr(extent, min_attr, new_val)
                    else:
                        warnings.warn(
                            f"Appended observation on dimension '{dim}' ({obs_min}) is below explicit {min_attr} ({min_coord}).",
                            UserWarning,
                            stacklevel=2,
                        )

            curr_max = getattr(extent, max_attr, msgspec.UNSET)
            if curr_max not in (None, msgspec.UNSET):
                max_coord = _to_coord(curr_max)
                if obs_max > max_coord:
                    if update_bounds:
                        new_val = (
                            [obs_max.numerator, obs_max.denominator]
                            if is_fraction
                            else obs_max
                        )
                        setattr(extent, max_attr, new_val)
                    else:
                        warnings.warn(
                            f"Appended observation on dimension '{dim}' ({obs_max}) exceeds explicit {max_attr} ({max_coord}).",
                            UserWarning,
                            stacklevel=2,
                        )

        # Disjoint check against preceding observation
        if getattr(extent, "disjoint", False) is True and n_obs > 0:
            if extent_tag == "time_interval":
                prev_end = extent.time[-1] + extent.duration[-1]
                if obs["time"] < prev_end:
                    warnings.warn(
                        f"Appended observation (time={obs['time']}) overlaps with preceding observation (end={prev_end}), violating disjoint=True.",
                        UserWarning,
                        stacklevel=2,
                    )
            elif extent_tag == "time":
                if obs["time"] <= extent.time[-1]:
                    warnings.warn(
                        f"Appended observation (time={obs['time']}) is not strictly after preceding observation (time={extent.time[-1]}), violating disjoint=True.",
                        UserWarning,
                        stacklevel=2,
                    )
            elif extent_tag == "score_interval":
                prev_end = _to_coord(extent.quarter[-1]) + _to_coord(
                    extent.duration[-1]
                )
                cur_start = _to_coord(obs["quarter"])
                if cur_start < prev_end:
                    warnings.warn(
                        f"Appended observation (quarter={cur_start}) overlaps with preceding observation (end={prev_end}), violating disjoint=True.",
                        UserWarning,
                        stacklevel=2,
                    )
            elif extent_tag == "score_quarter":
                prev_start = _to_coord(extent.quarter[-1])
                cur_start = _to_coord(obs["quarter"])
                if cur_start <= prev_start:
                    warnings.warn(
                        f"Appended observation (quarter={cur_start}) is not strictly after preceding observation (quarter={prev_start}), violating disjoint=True.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Exhaustiveness check
        if getattr(extent, "exhaustive", False) is True and n_obs > 0:
            if extent_tag == "time_interval":
                prev_end = extent.time[-1] + extent.duration[-1]
                if obs["time"] > prev_end:
                    warnings.warn(
                        f"Gap detected before appended observation (prev_end={prev_end}, cur_start={obs['time']}), violating exhaustive=True.",
                        UserWarning,
                        stacklevel=2,
                    )
            elif extent_tag == "score_interval":
                prev_end = _to_coord(extent.quarter[-1]) + _to_coord(
                    extent.duration[-1]
                )
                cur_start = _to_coord(obs["quarter"])
                if cur_start > prev_end:
                    warnings.warn(
                        f"Gap detected before appended observation (prev_end={prev_end}, cur_start={cur_start}), violating exhaustive=True.",
                        UserWarning,
                        stacklevel=2,
                    )

    # Grouping layer disjoint check against preceding observation of same group
    for g, gid in resolved_gids:
        if (
            getattr(g, "disjoint", False) is True
            and extent is not None
            and extent_tag == "time_interval"
        ):
            for idx in range(len(g.ids) - 1, -1, -1):
                if g.ids[idx] == gid:
                    prev_end = extent.time[idx] + extent.duration[idx]
                    if obs["time"] < prev_end:
                        warnings.warn(
                            f"Appended observation in grouping '{g.name}' (group {gid}) overlaps with previous group member (end={prev_end}), violating disjoint=True.",
                            UserWarning,
                            stacklevel=2,
                        )
                    break

    for fname in payload_cols:
        payload_cols[fname].append(obs[fname])
    for fname in extent_cols:
        extent_cols[fname].append(obs[fname])
    for fname in confidence_cols:
        confidence_cols[fname].append(obs[fname])
    for g, gid in resolved_gids:
        g.ids.append(gid)

    if inplace:
        annotation.id = msgspec.UNSET
        return annotation

    # Out-of-place derivation
    new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Appending to")
    target.parents = new_parents
    target.sandbox = new_sandbox

    if derive:
        Derived_cls = get_metadata_class("derived", version=annotation.bopp_version)
        target.metadata = Derived_cls(
            transform="append",
            parameters={"observation_index": n_obs},
            software=f"bopp=={__version__}",
        )

    target.id = msgspec.UNSET
    ensure_annotation_id(target)
    return target


def extend(
    annotation: BoppBase,
    observations: BoppBase | Sequence[dict[str, Any]] | dict[str, Sequence[Any]] | Any,
    *,
    inplace: bool = False,
    update_bounds: bool = False,
    derive: bool = True,
) -> BoppBase:
    """
    Extend an Annotation with multiple observations.

    Parameters
    ----------
    annotation : BoppBase
        The Annotation struct instance to extend.
    observations : BoppBase, sequence of dict, dict of sequences, or DataFrame
        Observations to append. Can be:
        1. Another compatible Annotation instance (matching media_id, version, facet types).
        2. A sequence of observation dictionaries.
        3. A dictionary of column lists.
        4. A pandas or polars DataFrame with compatible column names.
    inplace : bool, default False
        If True, mutates arrays in place and voids `annotation.id`.
        If False, returns a new derived Annotation with recomputed ID.
    update_bounds : bool, default False
        Whether to expand explicit extent bounds to encompass new observations.
    derive : bool, default True
        Whether to set metadata to DerivedAnnotationMetadata when inplace=False.

    Returns
    -------
    BoppBase
        The extended annotation (either mutated or newly derived).

    Raises
    ------
    BoppArgumentError
        If operands are incompatible or observation formats are invalid.
    BoppArrayError
        If columnar observation dictionaries contain mismatched lengths.
    """
    # 1. Compatibility check if observations is another Annotation
    if isinstance(observations, msgspec.Struct) and hasattr(observations, "media_id"):
        if observations.media_id != annotation.media_id:
            raise BoppArgumentError(
                f"media_id mismatch: '{observations.media_id}' != '{annotation.media_id}'."
            )
        if observations.bopp_version != annotation.bopp_version:
            raise BoppArgumentError(
                f"bopp_version mismatch: '{observations.bopp_version}' != '{annotation.bopp_version}'."
            )

        p1 = getattr(annotation, "payload", None)
        p2 = getattr(observations, "payload", None)
        if _get_tag(p1) != _get_tag(p2):
            raise BoppArgumentError(
                f"Payload type mismatch: '{_get_tag(p1)}' != '{_get_tag(p2)}'."
            )

        e1 = getattr(annotation, "extent", None)
        e2 = getattr(observations, "extent", None)
        t1 = _get_tag(e1) if e1 is not None else None
        t2 = _get_tag(e2) if e2 is not None else None
        if t1 != t2:
            raise BoppArgumentError(f"Extent type mismatch: '{t1}' != '{t2}'.")

        c1 = getattr(annotation, "confidence", None)
        c2 = getattr(observations, "confidence", None)
        ct1 = _get_tag(c1) if c1 is not None else None
        ct2 = _get_tag(c2) if c2 is not None else None
        if ct1 != ct2:
            raise BoppArgumentError(f"Confidence type mismatch: '{ct1}' != '{ct2}'.")

        g1_raw = getattr(annotation, "groupings", None)
        g1 = [] if g1_raw in (None, msgspec.UNSET) else list(g1_raw)
        g2_raw = getattr(observations, "groupings", None)
        g2 = [] if g2_raw in (None, msgspec.UNSET) else list(g2_raw)
        if len(g1) != len(g2):
            raise BoppArgumentError(
                f"Grouping layer count mismatch: {len(g1)} != {len(g2)}."
            )
        for idx in range(len(g1)):
            if g1[idx].name != g2[idx].name:
                raise BoppArgumentError(
                    f"Grouping name mismatch at index {idx}: '{g1[idx].name}' != '{g2[idx].name}'."
                )

        target = annotation if inplace else copy.deepcopy(annotation)
        (
            p_cols_target,
            e_cols_target,
            c_cols_target,
            g_target,
        ) = _get_annotation_columns(target)
        p_cols_src, e_cols_src, c_cols_src, g_src = _get_annotation_columns(
            observations
        )
        src_len = len(next(iter(p_cols_src.values()), []))
        n_obs_target = len(next(iter(p_cols_target.values()), []))

        if p1 is not None and p1 is not msgspec.UNSET:
            for fname in p1.__struct_fields__:
                v1 = getattr(p1, fname)
                v2 = getattr(p2, fname)
                if not isinstance(v1, list) and not isinstance(v2, list) and v1 != v2:
                    raise BoppArgumentError(
                        f"Scalar payload field '{fname}' mismatch: {v1!r} != {v2!r}."
                    )

        if c1 is not None and c1 is not msgspec.UNSET:
            for fname in c1.__struct_fields__:
                v1 = getattr(c1, fname)
                v2 = getattr(c2, fname)
                if not isinstance(v1, list) and not isinstance(v2, list):
                    if n_obs_target == 0 and v1 is msgspec.UNSET:
                        setattr(target.confidence, fname, v2)
                    elif v1 != v2:
                        raise BoppArgumentError(
                            f"Scalar confidence field '{fname}' mismatch: {v1!r} != {v2!r}."
                        )

        if e1 is not None and e1 is not msgspec.UNSET:
            if t1 == "time_uniform":
                if e1.sampling_period != e2.sampling_period:
                    raise BoppArgumentError(
                        f"Uniform extent sampling_period mismatch: {e1.sampling_period} != {e2.sampling_period}."
                    )
                if n_obs_target > 0 and src_len > 0:
                    expected_start = float(e1.start_time) + n_obs_target * float(
                        e1.sampling_period
                    )
                    if abs(float(e2.start_time) - expected_start) > 1e-9:
                        raise BoppArgumentError(
                            f"Uniform extent start_time mismatch: expected continuation at {expected_start}, got {e2.start_time}."
                        )
            if n_obs_target > 0:
                if e1.disjoint != e2.disjoint:
                    raise BoppArgumentError(
                        f"Extent disjoint setting mismatch: {e1.disjoint} != {e2.disjoint}."
                    )
                if e1.exhaustive != e2.exhaustive:
                    raise BoppArgumentError(
                        f"Extent exhaustive setting mismatch: {e1.exhaustive} != {e2.exhaustive}."
                    )
            elif src_len > 0:
                if t1 == "time_uniform":
                    target.extent.start_time = e2.start_time
                    target.extent.sampling_period = e2.sampling_period
                target.extent.disjoint = e2.disjoint
                target.extent.exhaustive = e2.exhaustive

        if n_obs_target > 0:
            for idx in range(len(g1)):
                if g1[idx].disjoint != g2[idx].disjoint:
                    raise BoppArgumentError(
                        f"Grouping '{g1[idx].name}' disjoint setting mismatch: {g1[idx].disjoint} != {g2[idx].disjoint}."
                    )
                if g1[idx].exhaustive != g2[idx].exhaustive:
                    raise BoppArgumentError(
                        f"Grouping '{g1[idx].name}' exhaustive setting mismatch: {g1[idx].exhaustive} != {g2[idx].exhaustive}."
                    )
        elif src_len > 0:
            for idx in range(len(g_target)):
                g_target[idx].disjoint = g2[idx].disjoint
                g_target[idx].exhaustive = g2[idx].exhaustive

        # Reconcile optional columns for empty target
        if n_obs_target == 0:
            for fname in p_cols_src:
                if fname not in p_cols_target:
                    setattr(target.payload, fname, [])
                    p_cols_target[fname] = getattr(target.payload, fname)
            if target.confidence is not None and target.confidence is not msgspec.UNSET:
                for fname in c_cols_src:
                    if fname not in c_cols_target:
                        setattr(target.confidence, fname, [])
                        c_cols_target[fname] = getattr(target.confidence, fname)

        if p_cols_target.keys() != p_cols_src.keys():
            raise BoppArgumentError(
                f"Payload column mismatch: target columns {sorted(p_cols_target.keys())} != source columns {sorted(p_cols_src.keys())}."
            )
        if e_cols_target.keys() != e_cols_src.keys():
            raise BoppArgumentError(
                f"Extent column mismatch: target columns {sorted(e_cols_target.keys())} != source columns {sorted(e_cols_src.keys())}."
            )
        if c_cols_target.keys() != c_cols_src.keys():
            raise BoppArgumentError(
                f"Confidence column mismatch: target columns {sorted(c_cols_target.keys())} != source columns {sorted(c_cols_src.keys())}."
            )

        # Validate and prepare grouping label merge and ID remapping before mutating arrays
        merged_groupings_data: list[
            tuple[dict[str, int] | msgspec.UnsetType, list[int]]
        ] = []
        for idx in range(len(g_target)):
            gt = g_target[idx]
            gs = g_src[idx]
            lbls_t = getattr(gt, "labels", msgspec.UNSET)
            lbls_s = getattr(gs, "labels", msgspec.UNSET)

            if lbls_s not in (None, msgspec.UNSET) and isinstance(lbls_s, dict):
                if lbls_t in (None, msgspec.UNSET):
                    merged_labels: dict[str, int] | msgspec.UnsetType = dict(lbls_s)
                    remapped_ids = list(gs.ids)
                else:
                    merged_labels = dict(lbls_t)
                    used_ids = set(merged_labels.values()) | {
                        i for i in gt.ids if i >= 0
                    }
                    id_remap: dict[int, int] = {}
                    for label_name, src_id in lbls_s.items():
                        if label_name in merged_labels:
                            tgt_id = merged_labels[label_name]
                            if src_id != tgt_id:
                                id_remap[src_id] = tgt_id
                        else:
                            if src_id not in used_ids:
                                merged_labels[label_name] = src_id
                                used_ids.add(src_id)
                            else:
                                new_id = max(used_ids, default=-1) + 1
                                merged_labels[label_name] = new_id
                                used_ids.add(new_id)
                                id_remap[src_id] = new_id
                    remapped_ids = [id_remap.get(x, x) for x in gs.ids]
            else:
                merged_labels = dict(lbls_t) if isinstance(lbls_t, dict) else lbls_t
                remapped_ids = list(gs.ids)

            merged_groupings_data.append((merged_labels, remapped_ids))

        # Bounds check / update on target extent
        target_extent = getattr(target, "extent", None)
        target_extent_tag = (
            _get_tag(target_extent) if target_extent is not None else None
        )
        if (
            target_extent is not None
            and target_extent is not msgspec.UNSET
            and src_len > 0
        ):
            specs = EXTENT_AXES_METADATA.get(target_extent_tag or "", [])
            is_fraction = target_extent_tag in FRACTION_EXTENT_TAGS
            if target_extent_tag == "time_uniform":
                st = float(target_extent.start_time)
                sp = float(target_extent.sampling_period)
                obs_bounds = {
                    "time": (
                        st + n_obs_target * sp,
                        st + (n_obs_target + src_len - 1) * sp,
                    )
                }
            else:
                obs_bounds = infer_extent_bounds(observations.extent, N=src_len)

            for spec in specs:
                dim = spec["dim"]
                min_attr = spec["min_attr"]
                max_attr = spec["max_attr"]
                inf_min, inf_max = obs_bounds[dim]
                curr_min = getattr(target_extent, min_attr, msgspec.UNSET)
                if curr_min not in (None, msgspec.UNSET):
                    min_coord = _to_coord(curr_min)
                    if inf_min < min_coord:
                        if update_bounds:
                            new_val = (
                                [inf_min.numerator, inf_min.denominator]
                                if is_fraction
                                else inf_min
                            )
                            setattr(target_extent, min_attr, new_val)
                        else:
                            warnings.warn(
                                f"Extended observation on dimension '{dim}' ({inf_min}) is below explicit {min_attr} ({min_coord}).",
                                UserWarning,
                                stacklevel=2,
                            )

                curr_max = getattr(target_extent, max_attr, msgspec.UNSET)
                if curr_max not in (None, msgspec.UNSET):
                    max_coord = _to_coord(curr_max)
                    if inf_max > max_coord:
                        if update_bounds:
                            new_val = (
                                [inf_max.numerator, inf_max.denominator]
                                if is_fraction
                                else inf_max
                            )
                            setattr(target_extent, max_attr, new_val)
                        else:
                            warnings.warn(
                                f"Extended observation on dimension '{dim}' ({inf_max}) exceeds explicit {max_attr} ({max_coord}).",
                                UserWarning,
                                stacklevel=2,
                            )

            # Disjoint sanity check
            if n_obs_target > 0 and getattr(target_extent, "disjoint", False) is True:
                if target_extent_tag == "time_interval":
                    prev_end = target_extent.time[-1] + target_extent.duration[-1]
                    cur_start = observations.extent.time[0]
                    if cur_start < prev_end:
                        warnings.warn(
                            f"Extended observation (time={cur_start}) overlaps with preceding observation (end={prev_end}), violating disjoint=True.",
                            UserWarning,
                            stacklevel=2,
                        )
                elif target_extent_tag == "time":
                    prev_time = target_extent.time[-1]
                    cur_time = observations.extent.time[0]
                    if cur_time <= prev_time:
                        warnings.warn(
                            f"Extended observation (time={cur_time}) is not strictly after preceding observation (time={prev_time}), violating disjoint=True.",
                            UserWarning,
                            stacklevel=2,
                        )
                elif target_extent_tag == "score_interval":
                    prev_end = _to_coord(target_extent.quarter[-1]) + _to_coord(
                        target_extent.duration[-1]
                    )
                    cur_start = _to_coord(observations.extent.quarter[0])
                    if cur_start < prev_end:
                        warnings.warn(
                            f"Extended observation (quarter={cur_start}) overlaps with preceding observation (end={prev_end}), violating disjoint=True.",
                            UserWarning,
                            stacklevel=2,
                        )
                elif target_extent_tag == "score_quarter":
                    prev_start = _to_coord(target_extent.quarter[-1])
                    cur_start = _to_coord(observations.extent.quarter[0])
                    if cur_start <= prev_start:
                        warnings.warn(
                            f"Extended observation (quarter={cur_start}) is not strictly after preceding observation (quarter={prev_start}), violating disjoint=True.",
                            UserWarning,
                            stacklevel=2,
                        )

            # Exhaustiveness check
            if n_obs_target > 0 and getattr(target_extent, "exhaustive", False) is True:
                if target_extent_tag == "time_interval":
                    prev_end = target_extent.time[-1] + target_extent.duration[-1]
                    cur_start = observations.extent.time[0]
                    if cur_start > prev_end:
                        warnings.warn(
                            f"Gap detected before extended observation (prev_end={prev_end}, cur_start={cur_start}), violating exhaustive=True.",
                            UserWarning,
                            stacklevel=2,
                        )
                elif target_extent_tag == "score_interval":
                    prev_end = _to_coord(target_extent.quarter[-1]) + _to_coord(
                        target_extent.duration[-1]
                    )
                    cur_start = _to_coord(observations.extent.quarter[0])
                    if cur_start > prev_end:
                        warnings.warn(
                            f"Gap detected before extended observation (prev_end={prev_end}, cur_start={cur_start}), violating exhaustive=True.",
                            UserWarning,
                            stacklevel=2,
                        )

            # Grouping layer disjoint check
            if n_obs_target > 0:
                for idx in range(len(g_target)):
                    gt = g_target[idx]
                    if (
                        getattr(gt, "disjoint", False) is True
                        and target_extent_tag == "time_interval"
                    ):
                        last_end: dict[int, float] = {}
                        for k in range(n_obs_target):
                            gid_k = gt.ids[k]
                            end_k = target_extent.time[k] + target_extent.duration[k]
                            last_end[gid_k] = end_k
                        remap_ids = merged_groupings_data[idx][1]
                        for j in range(src_len):
                            gid_j = remap_ids[j]
                            cur_t = observations.extent.time[j]
                            if gid_j in last_end and cur_t < last_end[gid_j]:
                                warnings.warn(
                                    f"Extended observation in grouping '{gt.name}' (group {gid_j}) overlaps with previous group member (end={last_end[gid_j]}), violating disjoint=True.",
                                    UserWarning,
                                    stacklevel=2,
                                )
                                break
                            last_end[gid_j] = cur_t + observations.extent.duration[j]

        # Commit observation rows to target arrays
        for fname in p_cols_target:
            p_cols_target[fname].extend(p_cols_src[fname])
        for fname in e_cols_target:
            e_cols_target[fname].extend(e_cols_src[fname])
        for fname in c_cols_target:
            c_cols_target[fname].extend(c_cols_src[fname])

        for idx, (m_labels, remap_ids) in enumerate(merged_groupings_data):
            g_target[idx].labels = m_labels
            g_target[idx].ids.extend(remap_ids)

        if inplace:
            annotation.id = msgspec.UNSET
            return annotation

        new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Extending")
        if getattr(observations, "id", msgspec.UNSET) not in (None, msgspec.UNSET):
            new_parents.append(observations.id)
        target.parents = new_parents
        target.sandbox = new_sandbox
        if derive:
            Derived_cls = get_metadata_class("derived", version=annotation.bopp_version)
            target.metadata = Derived_cls(
                transform="extend",
                parameters={"count": src_len},
                software=f"bopp=={__version__}",
            )
        target.id = msgspec.UNSET
        ensure_annotation_id(target)
        return target

    # 2. DataFrame conversion
    if hasattr(observations, "to_dict"):
        try:
            observations = observations.to_dict(orient="list")
        except TypeError:
            observations = observations.to_dict(as_series=False)

    # 3. Columnar dictionary conversion
    if isinstance(observations, dict):
        lens = [
            len(v)
            for v in observations.values()
            if isinstance(v, (list, tuple, Sequence))
            and not isinstance(v, (str, bytes))
        ]
        if len(set(lens)) > 1:
            raise BoppArrayError(
                f"Mismatched array lengths in observations dict: {set(lens)}"
            )
        K = lens[0] if lens else 0
        observations = [{k: observations[k][i] for k in observations} for i in range(K)]

    # 4. Sequence of observation dicts
    if isinstance(observations, Sequence) and not isinstance(
        observations, (str, bytes)
    ):
        if inplace:
            for obs_item in observations:
                append(annotation, obs_item, inplace=True, update_bounds=update_bounds)
            return annotation

        target = copy.deepcopy(annotation)
        for obs_item in observations:
            append(target, obs_item, inplace=True, update_bounds=update_bounds)
        new_parents, new_sandbox = _derive_parents_and_sandbox(annotation, "Extending")
        target.parents = new_parents
        target.sandbox = new_sandbox
        if derive:
            Derived_cls = get_metadata_class("derived", version=annotation.bopp_version)
            target.metadata = Derived_cls(
                transform="extend",
                parameters={"count": len(observations)},
                software=f"bopp=={__version__}",
            )
        target.id = msgspec.UNSET
        ensure_annotation_id(target)
        return target

    raise BoppArgumentError(
        f"Unsupported observations type for extend: {type(observations).__name__}."
    )


T = TypeVar("T")


def validate(obj: Any, target_type: type[T] | None = None) -> bool:
    """
    Validate an object against a target msgspec structure type.

    Parameters
    ----------
    obj : Any
        Object to validate (e.g., a msgspec Struct instance or a dict).
    target_type : type of T or None, optional
        Target struct type to validate against. If None and `obj` is a
        `msgspec.Struct`, `type(obj)` is used.

    Returns
    -------
    bool
        True if validation succeeds.

    Raises
    ------
    BoppArgumentError
        If `target_type` is missing and cannot be inferred.
    BoppValidationError
        If validation fails.
    """
    if target_type is None:
        if isinstance(obj, msgspec.Struct):
            target_type = cast(type[T], type(obj))
        else:
            raise BoppArgumentError(
                "target_type must be provided if obj is not a msgspec.Struct."
            )

    # 1. Handle Struct instances: convert to builtins to strip UNSET keys recursively
    if isinstance(obj, msgspec.Struct):
        obj = msgspec.to_builtins(obj)
    # 2. Handle raw dicts: filter out top-level UNSET sentinels if manually populated
    elif isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if v is not msgspec.UNSET}

    try:
        msgspec.convert(obj, target_type)
    except (msgspec.ValidationError, TypeError) as err:
        raise BoppValidationError(
            f"Validation failed for {target_type.__name__}: {err}"
        ) from err
    return True
