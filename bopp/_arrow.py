"""Apache Arrow array integration, validation, and msgspec codec hooks for BOPP."""

from __future__ import annotations

import functools
import re
from enum import Enum
from typing import (
    Any,
    Generic,
    TypeVar,
    get_args,
    get_origin,
)

import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

DType = TypeVar("DType")
T = TypeVar("T")

DTYPE_STR_TO_PA: dict[str, pa.DataType] = {
    "float32": pa.float32(),
    "float64": pa.float64(),
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "int32": pa.int32(),
    "bool": pa.bool_(),
    "string": pa.string(),
}


class _ArrowMeta(type):
    """Metaclass allowing isinstance(instance, ArrowArray) to match pyarrow Array instances."""

    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, (pa.Array, pa.ChunkedArray))


class ArrowArray(Generic[DType, T], metaclass=_ArrowMeta):
    """Generic marker type for columnar Apache Arrow arrays in BOPP msgspec models."""


def unwrap_arrow_array(tp: Any) -> tuple[str, Any] | None:
    """Extract (dtype_name, item_type) from an ArrowArray type annotation.

    Parameters
    ----------
    tp : Any
        Type annotation to unwrap.

    Returns
    -------
    tuple of (str, Any) or None
        A tuple of (dtype_name, item_type) if `tp` is an ArrowArray, else None.
    """
    visited = set()
    curr = tp

    while curr is not None and id(curr) not in visited:
        visited.add(id(curr))

        # Check for PEP 695 TypeAliasType
        if hasattr(curr, "__value__"):
            curr = curr.__value__
            continue

        origin = get_origin(curr)
        if getattr(origin, "__name__", "") == "ArrowArray":
            args = get_args(curr)
            if len(args) >= 2:
                dtype_arg = args[0]
                dtype_name = (
                    get_args(dtype_arg)[0]
                    if get_origin(dtype_arg) is not None
                    else str(dtype_arg)
                )
                if isinstance(dtype_name, str):
                    dtype_name = dtype_name.strip("'\"")
                return str(dtype_name), args[1]
            break

        # Unwrap Annotated[T, ...]
        if hasattr(curr, "__metadata__"):
            sub_args = get_args(curr)
            curr = sub_args[0] if sub_args else None
            continue

        # Unwrap Optional[T] / Union[T, None, UnsetType]
        if origin is not None and hasattr(curr, "__args__"):
            sub_args = [
                a
                for a in get_args(curr)
                if a not in (type(None), msgspec.UnsetType)
            ]
            if len(sub_args) == 1:
                curr = sub_args[0]
                continue

        break

    return None


class _ItemConstraints:
    """Cached validation metadata for ArrowArray item types."""

    __slots__ = (
        "enum_values",
        "ge",
        "gt",
        "has_numeric_bounds",
        "le",
        "lt",
        "pattern",
        "re_pattern",
    )

    def __init__(
        self,
        *,
        ge: float | int | None = None,
        gt: float | int | None = None,
        le: float | int | None = None,
        lt: float | int | None = None,
        pattern: str | None = None,
        re_pattern: re.Pattern | None = None,
        enum_values: set[Any] | None = None,
    ) -> None:
        self.ge = ge
        self.gt = gt
        self.le = le
        self.lt = lt
        self.has_numeric_bounds = any(
            v is not None for v in (ge, gt, le, lt)
        )
        self.pattern = pattern
        self.re_pattern = re_pattern
        self.enum_values = enum_values


@functools.cache
def _get_item_constraints(item_type: Any) -> _ItemConstraints:
    ge: float | int | None = None
    gt: float | int | None = None
    le: float | int | None = None
    lt: float | int | None = None
    pattern: str | None = None
    enum_values: set[Any] | None = None

    curr = item_type
    while curr is not None:
        if hasattr(curr, "__value__"):
            curr = curr.__value__
            continue

        if hasattr(curr, "__metadata__"):
            for meta in curr.__metadata__:
                if isinstance(meta, msgspec.Meta):
                    if meta.ge is not None:
                        ge = meta.ge if ge is None else max(ge, meta.ge)
                    if meta.gt is not None:
                        gt = meta.gt if gt is None else max(gt, meta.gt)
                    if meta.le is not None:
                        le = meta.le if le is None else min(le, meta.le)
                    if meta.lt is not None:
                        lt = meta.lt if lt is None else min(lt, meta.lt)
                    if meta.pattern is not None:
                        pattern = meta.pattern
            sub_args = get_args(curr)
            curr = sub_args[0] if sub_args else None
            continue

        if isinstance(curr, type) and issubclass(curr, Enum):
            enum_values = {e.value for e in curr}
            break

        break

    re_pattern = re.compile(pattern) if pattern is not None else None
    return _ItemConstraints(
        ge=ge,
        gt=gt,
        le=le,
        lt=lt,
        pattern=pattern,
        re_pattern=re_pattern,
        enum_values=enum_values,
    )


def coerce_arrow_array(
    obj: Any,
    dtype_name: str,
    item_type: Any = None,
    *,
    validate_constraints: bool = True,
) -> pa.Array:
    """Coerce input into a validated pyarrow Array conforming to dtype and item constraints.

    Parameters
    ----------
    obj : Any
        Container or array object to coerce (list, tuple, numpy array, polars Series, etc.).
    dtype_name : str
        Target Arrow data type name (e.g. 'float32', 'uint8').
    item_type : Any, optional
        Annotated item type carrying validation metadata, by default None.
    validate_constraints : bool, default=True
        Whether to run vectorized validation checks on the elements.

    Returns
    -------
    pa.Array
        Coerced and validated PyArrow array.
    """
    target_pa_type = DTYPE_STR_TO_PA.get(dtype_name)
    if target_pa_type is None:
        raise msgspec.ValidationError(f"Unknown Arrow dtype '{dtype_name}'")

    if isinstance(obj, (pa.Array, pa.ChunkedArray)):
        if isinstance(obj, pa.ChunkedArray):
            obj = obj.combine_chunks()
        if obj.type != target_pa_type:
            try:
                obj = obj.cast(target_pa_type)
            except Exception as exc:
                raise msgspec.ValidationError(
                    f"Cannot cast {obj.type} to {target_pa_type}: {exc}"
                ) from exc
    elif hasattr(obj, "to_arrow"):
        # Handles polars.Series
        arr = obj.to_arrow()
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        if arr.type != target_pa_type:
            arr = arr.cast(target_pa_type)
        obj = arr
    elif isinstance(obj, np.ndarray):
        try:
            obj = pa.array(obj, type=target_pa_type)
        except Exception as exc:
            raise msgspec.ValidationError(
                f"Failed to convert numpy array to {target_pa_type}: {exc}"
            ) from exc
    elif isinstance(obj, (list, tuple)):
        # Strict scalar type validation matching msgspec semantics
        if dtype_name in ("uint8", "uint16", "uint32", "int32"):
            for x in obj:
                if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
                    raise msgspec.ValidationError(
                        f"Expected integer for {dtype_name} array, got {type(x).__name__}"
                    )
        elif dtype_name in ("float32", "float64"):
            for x in obj:
                if isinstance(x, bool) or not isinstance(
                    x, (int, float, np.floating, np.integer)
                ):
                    raise msgspec.ValidationError(
                        f"Expected float for {dtype_name} array, got {type(x).__name__}"
                    )
        elif dtype_name == "bool":
            for x in obj:
                if not isinstance(x, (bool, np.bool_)):
                    raise msgspec.ValidationError(
                        f"Expected bool for bool array, got {type(x).__name__}"
                    )
        elif dtype_name == "string":
            for x in obj:
                if not isinstance(x, str):
                    raise msgspec.ValidationError(
                        f"Expected str for string array, got {type(x).__name__}"
                    )
        try:
            obj = pa.array(obj, type=target_pa_type)
        except Exception as exc:
            raise msgspec.ValidationError(
                f"Failed to create Arrow array of type {target_pa_type}: {exc}"
            ) from exc
    elif type(obj).__name__ == "Series":  # pandas Series
        try:
            obj = pa.array(obj, type=target_pa_type)
        except Exception as exc:
            raise msgspec.ValidationError(
                f"Failed to convert pandas Series to {target_pa_type}: {exc}"
            ) from exc
    else:
        raise msgspec.ValidationError(
            f"Cannot coerce object of type {type(obj).__name__} to ArrowArray"
        )

    if obj.null_count > 0:
        raise msgspec.ValidationError("Null values not allowed in bopp arrays")

    if dtype_name in ("float32", "float64") and len(obj) > 0:
        if not pc.all(pc.is_finite(obj)).as_py():
            raise msgspec.ValidationError(
                "Float array contains non-finite values (NaN or Inf)"
            )

    if validate_constraints and item_type is not None and len(obj) > 0:
        constraints = _get_item_constraints(item_type)
        if constraints.has_numeric_bounds:
            mm = pc.min_max(obj)
            min_val = mm["min"].as_py()
            max_val = mm["max"].as_py()
            if constraints.ge is not None and min_val < constraints.ge:
                raise msgspec.ValidationError(
                    f"Array element {min_val} < minimum {constraints.ge}"
                )
            if constraints.gt is not None and min_val <= constraints.gt:
                raise msgspec.ValidationError(
                    f"Array element {min_val} <= exclusiveMinimum {constraints.gt}"
                )
            if constraints.le is not None and max_val > constraints.le:
                raise msgspec.ValidationError(
                    f"Array element {max_val} > maximum {constraints.le}"
                )
            if constraints.lt is not None and max_val >= constraints.lt:
                raise msgspec.ValidationError(
                    f"Array element {max_val} >= exclusiveMaximum {constraints.lt}"
                )

        if constraints.re_pattern is not None:
            uniques = pc.unique(obj).to_pylist()
            for u in uniques:
                if not constraints.re_pattern.fullmatch(u):
                    raise msgspec.ValidationError(
                        f"String '{u}' does not match pattern '{constraints.pattern}'"
                    )

        if constraints.enum_values is not None:
            uniques = pc.unique(obj).to_pylist()
            for u in uniques:
                if u not in constraints.enum_values:
                    raise msgspec.ValidationError(
                        f"Value '{u}' not in enum {sorted(constraints.enum_values)}"
                    )

    return obj


def arrow_dec_hook(tp: Any, obj: Any) -> Any:
    """Msgspec decode hook handling ArrowArray fields.

    Parameters
    ----------
    tp : Any
        Target type being decoded into.
    obj : Any
        Object being decoded.

    Returns
    -------
    Any
        Decoded PyArrow array or Python list.
    """
    unwrapped = unwrap_arrow_array(tp)
    if unwrapped is not None:
        dtype_name, item_type = unwrapped
        return coerce_arrow_array(obj, dtype_name, item_type)
    if isinstance(obj, (pa.Array, pa.ChunkedArray)):
        return obj.to_pylist()
    raise TypeError(f"Cannot decode {obj!r} to {tp}")


def arrow_json_enc_hook(obj: Any) -> Any:
    """Msgspec JSON encoding hook formatting Arrow arrays with clean float32 representation.

    Parameters
    ----------
    obj : Any
        Object being encoded to JSON.

    Returns
    -------
    Any
        JSON-serializable Python list or primitive representation.
    """
    if isinstance(obj, (pa.Array, pa.ChunkedArray)):
        if isinstance(obj, pa.ChunkedArray):
            obj = obj.combine_chunks()
        if obj.type == pa.float32():
            np_arr = obj.to_numpy(zero_copy_only=False)
            return [float(f"{x:g}") for x in np_arr]
        return obj.to_pylist()
    raise TypeError(f"Objects of type {type(obj)} are not JSON serializable")


def arrow_msgpack_enc_hook(obj: Any) -> Any:
    """Msgspec MessagePack encoding hook packing compact float32 elements vectorized via msgspec.Raw.

    Parameters
    ----------
    obj : Any
        Object being encoded to MessagePack.

    Returns
    -------
    Any
        Msgspec Raw instance or MessagePack-compatible representation.
    """
    if isinstance(obj, (pa.Array, pa.ChunkedArray)):
        if isinstance(obj, pa.ChunkedArray):
            obj = obj.combine_chunks()
        n = len(obj)
        if obj.type == pa.float32():
            np_arr = obj.to_numpy(zero_copy_only=False)
            struct_dtype = np.dtype([("tag", "u1"), ("val", ">f4")])
            packed = np.empty(n, dtype=struct_dtype)
            packed["tag"] = 0xCA
            packed["val"] = np_arr
            header = (
                bytes([0x90 | n])
                if n < 16
                else (
                    b"\xdc" + n.to_bytes(2, "big")
                    if n < 65536
                    else b"\xdd" + n.to_bytes(4, "big")
                )
            )
            return msgspec.Raw(header + packed.tobytes())
        if obj.type == pa.float64():
            np_arr = obj.to_numpy(zero_copy_only=False)
            struct_dtype = np.dtype([("tag", "u1"), ("val", ">f8")])
            packed = np.empty(n, dtype=struct_dtype)
            packed["tag"] = 0xCB
            packed["val"] = np_arr
            header = (
                bytes([0x90 | n])
                if n < 16
                else (
                    b"\xdc" + n.to_bytes(2, "big")
                    if n < 65536
                    else b"\xdd" + n.to_bytes(4, "big")
                )
            )
            return msgspec.Raw(header + packed.tobytes())
        return obj.to_pylist()
    raise TypeError(f"Objects of type {type(obj)} are not MessagePack serializable")
