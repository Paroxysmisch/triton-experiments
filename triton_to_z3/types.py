"""MLIR type parsing and Z3 sort mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass

import z3


# ---------------------------------------------------------------------------
# MLIR type representations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntegerType:
    width: int  # 1, 8, 16, 32, 64 ...

    def to_z3_sort(self) -> z3.SortRef:
        if self.width == 1:
            return z3.BoolSort()
        return z3.BitVecSort(self.width)


@dataclass(frozen=True)
class FloatType:
    width: int  # 16, 32, 64

    def to_z3_sort(self) -> z3.FPSortRef:
        match self.width:
            case 16:
                return z3.Float16()
            case 32:
                return z3.Float32()
            case 64:
                return z3.Float64()
            case _:
                return z3.FPSort(5, self.width - 5)


@dataclass(frozen=True)
class BFloat16Type:
    def to_z3_sort(self) -> z3.FPSortRef:
        return z3.FPSort(8, 8)


@dataclass(frozen=True)
class IndexType:
    def to_z3_sort(self) -> z3.SortRef:
        return z3.BitVecSort(64)


@dataclass(frozen=True)
class PointerType:
    pointee: MLIRType

    def to_z3_sort(self) -> z3.SortRef:
        return z3.BitVecSort(64)


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    element_type: MLIRType

    def to_z3_sort(self) -> z3.SortRef:
        """For symbolic execution we treat tensors as their element type."""
        return self.element_type.to_z3_sort()


MLIRType = IntegerType | FloatType | BFloat16Type | IndexType | PointerType | TensorType


# ---------------------------------------------------------------------------
# Type parsing
# ---------------------------------------------------------------------------

def parse_type(s: str) -> MLIRType:
    """Parse an MLIR type string into an MLIRType."""
    s = s.strip()
    if not s:
        return FloatType(32)

    # tensor<shape x element_type>
    m = re.match(r"tensor<(.+)>", s)
    if m:
        inner = m.group(1)
        return _parse_tensor_inner(inner)

    # !tt.ptr<element_type>
    m = re.match(r"!tt\.ptr<(.+)>", s)
    if m:
        return PointerType(parse_type(m.group(1)))

    # integer types: i1, i8, i16, i32, i64
    m = re.match(r"i(\d+)$", s)
    if m:
        return IntegerType(int(m.group(1)))

    # float types: f16, f32, f64
    m = re.match(r"f(\d+)$", s)
    if m:
        return FloatType(int(m.group(1)))

    if s == "bf16":
        return BFloat16Type()

    if s == "index":
        return IndexType()

    # Fallback
    return FloatType(32)


def _parse_tensor_inner(inner: str) -> TensorType:
    """Parse the inside of tensor<...>, e.g. '1024xf32' or '1x1024x!tt.ptr<f32>'."""
    # Walk left to right, collecting numeric dimensions.
    # The element type starts at the first non-numeric-x segment.
    parts: list[str] = []
    rest = inner
    while rest:
        m = re.match(r"(\d+)x", rest)
        if m:
            parts.append(m.group(1))
            rest = rest[m.end():]
        else:
            break
    shape = tuple(int(p) for p in parts)
    elem = parse_type(rest) if rest else FloatType(32)
    return TensorType(shape, elem)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def element_type(t: MLIRType) -> MLIRType:
    """Unwrap tensor / pointer to the scalar element type."""
    match t:
        case TensorType(element_type=inner):
            return element_type(inner)
        case PointerType(pointee=inner):
            return inner
        case _:
            return t


def is_float_type(t: MLIRType) -> bool:
    match element_type(t):
        case FloatType() | BFloat16Type():
            return True
        case _:
            return False


def is_int_type(t: MLIRType) -> bool:
    match element_type(t):
        case IntegerType() | IndexType():
            return True
        case _:
            return False


def is_bool_type(t: MLIRType) -> bool:
    match element_type(t):
        case IntegerType(width=1):
            return True
        case _:
            return False


def z3_sort(t: MLIRType) -> z3.SortRef:
    """Get the Z3 sort for a type (unwraps tensors)."""
    return element_type(t).to_z3_sort()


def make_z3_var(name: str, t: MLIRType) -> z3.ExprRef:
    """Create a fresh Z3 symbolic variable for the given type."""
    scalar = element_type(t)
    sort = scalar.to_z3_sort()
    match scalar:
        case FloatType() | BFloat16Type():
            return z3.FP(name, sort)
        case IntegerType(width=1):
            return z3.Bool(name)
        case IntegerType(width=w):
            return z3.BitVec(name, w)
        case IndexType():
            return z3.BitVec(name, 64)
        case PointerType():
            return z3.BitVec(name, 64)
        case _:
            return z3.FP(name, z3.Float32())


def fp_sort_for_width(width: int) -> z3.FPSortRef:
    """Return the Z3 FP sort for a given bit-width."""
    match width:
        case 16:
            return z3.Float16()
        case 32:
            return z3.Float32()
        case 64:
            return z3.Float64()
        case _:
            return z3.FPSort(5, width - 5)


def bv_width(t: MLIRType) -> int:
    """Return the bit-vector width for an integer-like type."""
    match element_type(t):
        case IntegerType(width=w):
            return w
        case IndexType():
            return 64
        case PointerType():
            return 64
        case _:
            return 32
