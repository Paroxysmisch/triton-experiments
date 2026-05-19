from __future__ import annotations

from dataclasses import dataclass
from typing import Union


class UnsupportedIR(ValueError):
    """Raised when an IR module is outside the trusted fragment."""


@dataclass(frozen=True)
class IndexExpr:
    pass


@dataclass(frozen=True)
class IndexConst(IndexExpr):
    value: int


@dataclass(frozen=True)
class IndexSymbol(IndexExpr):
    name: str


@dataclass(frozen=True)
class IndexAdd(IndexExpr):
    left: IndexExpr
    right: IndexExpr


@dataclass(frozen=True)
class IndexMul(IndexExpr):
    left: IndexExpr
    right: IndexExpr


@dataclass(frozen=True)
class BoolExpr:
    pass


@dataclass(frozen=True)
class BoolLt(BoolExpr):
    left: IndexExpr
    right: IndexExpr


@dataclass(frozen=True)
class BoolLe(BoolExpr):
    left: IndexExpr
    right: IndexExpr


@dataclass(frozen=True)
class ValueExpr:
    pass


@dataclass(frozen=True)
class ValueConst(ValueExpr):
    value: str


@dataclass(frozen=True)
class ValueScalar(ValueExpr):
    name: str


@dataclass(frozen=True)
class ValueLoad(ValueExpr):
    tensor: str
    index: IndexExpr


@dataclass(frozen=True)
class ValueUnary(ValueExpr):
    op: str
    arg: ValueExpr


@dataclass(frozen=True)
class ValueBinary(ValueExpr):
    op: str
    left: ValueExpr
    right: ValueExpr


@dataclass(frozen=True)
class KernelSummary:
    fragment: str
    n_symbol: str
    block_symbol: str
    index: IndexExpr
    mask: BoolExpr
    output: str
    value: ValueExpr


Expr = Union[IndexExpr, BoolExpr, ValueExpr]


def idx_const(value: int) -> IndexConst:
    return IndexConst(value)


def simplify_index(expr: IndexExpr) -> IndexExpr:
    if isinstance(expr, IndexAdd):
        left = simplify_index(expr.left)
        right = simplify_index(expr.right)
        if isinstance(left, IndexConst) and left.value == 0:
            return right
        if isinstance(right, IndexConst) and right.value == 0:
            return left
        if isinstance(left, IndexConst) and isinstance(right, IndexConst):
            return IndexConst(left.value + right.value)
        # Canonicalize commutative add by repr for stable equality.
        if repr(right) < repr(left):
            left, right = right, left
        return IndexAdd(left, right)
    if isinstance(expr, IndexMul):
        left = simplify_index(expr.left)
        right = simplify_index(expr.right)
        if isinstance(left, IndexConst) and left.value == 0:
            return IndexConst(0)
        if isinstance(right, IndexConst) and right.value == 0:
            return IndexConst(0)
        if isinstance(left, IndexConst) and left.value == 1:
            return right
        if isinstance(right, IndexConst) and right.value == 1:
            return left
        if isinstance(left, IndexConst) and isinstance(right, IndexConst):
            return IndexConst(left.value * right.value)
        if repr(right) < repr(left):
            left, right = right, left
        return IndexMul(left, right)
    return expr


def canonical_index(block_symbol: str = "BLOCK") -> IndexExpr:
    return simplify_index(
        IndexAdd(
            IndexMul(IndexSymbol("pid0"), IndexSymbol(block_symbol)),
            IndexSymbol("lane"),
        )
    )

