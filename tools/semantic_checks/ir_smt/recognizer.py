from __future__ import annotations

import re

from .kernel_ir import (
    BoolExpr,
    BoolLe,
    BoolLt,
    IndexAdd,
    IndexConst,
    IndexExpr,
    IndexMul,
    IndexSymbol,
    KernelSummary,
    UnsupportedIR,
    ValueBinary,
    ValueConst,
    ValueExpr,
    ValueLoad,
    ValueScalar,
    ValueUnary,
    simplify_index,
)
from .ssa_parser import Module, Operation


INTEGER_OPS = {"arith.addi": "add", "arith.muli": "mul", "tt.addptr": "add"}
FLOAT_OPS = {
    "arith.addf": "+",
    "arith.subf": "-",
    "arith.mulf": "*",
    "arith.divf": "/",
}
UNARY_OPS = {
    "math.sqrt": "sqrt",
    "tt.sqrt": "sqrt",
    "math.tanh": "tanh",
    "tt.tanh": "tanh",
}
LOAD_OPS = {"tt.load", "triton_gpu.load"}
STORE_OPS = {"tt.store", "triton_gpu.store"}
PROGRAM_ID_OPS = {"tt.get_program_id", "triton_gpu.get_program_id"}
RANGE_OPS = {"tt.make_range", "tt.arange", "triton_gpu.make_range"}
CMP_OPS = {"arith.cmpi"}


def _const_from_raw(op: Operation) -> int | None:
    match = re.search(r"\b(-?[0-9]+)\b", op.raw)
    return int(match.group(1)) if match else None


def _symbol_for_unknown(value: str) -> IndexSymbol:
    return IndexSymbol(value.lstrip("%"))


def _def(module: Module, value: str) -> Operation | None:
    return module.defs.get(value)


def recognize_index(module: Module, value: str) -> IndexExpr:
    op = _def(module, value)
    if op is None:
        return _symbol_for_unknown(value)
    if op.op_name in PROGRAM_ID_OPS:
        axis = op.attrs.get("axis", "0")
        if axis not in {"0", "x"}:
            raise UnsupportedIR(f"only program_id axis 0 is supported: {op.raw}")
        return IndexSymbol("pid0")
    if op.op_name in RANGE_OPS:
        return IndexSymbol("lane")
    if op.op_name in {"arith.constant", "tt.const"}:
        const = _const_from_raw(op)
        if const is None:
            raise UnsupportedIR(f"unsupported non-integer constant in index: {op.raw}")
        return IndexConst(const)
    if op.op_name in INTEGER_OPS:
        if len(op.operands) < 2:
            raise UnsupportedIR(f"binary index op has too few operands: {op.raw}")
        left = recognize_index(module, op.operands[0])
        right = recognize_index(module, op.operands[1])
        if INTEGER_OPS[op.op_name] == "add":
            return simplify_index(IndexAdd(left, right))
        if INTEGER_OPS[op.op_name] == "mul":
            return simplify_index(IndexMul(left, right))
    raise UnsupportedIR(f"unsupported index op: {op.raw}")


def recognize_mask(module: Module, value: str) -> BoolExpr:
    op = _def(module, value)
    if op is None or op.op_name not in CMP_OPS:
        raise UnsupportedIR(f"unsupported mask value: {value}")
    if len(op.operands) < 2:
        raise UnsupportedIR(f"comparison has too few operands: {op.raw}")
    left = recognize_index(module, op.operands[0])
    right = recognize_index(module, op.operands[1])
    raw = op.raw.lower()
    if "slt" in raw or "ult" in raw or " lt" in raw:
        return BoolLt(left, right)
    if "sle" in raw or "ule" in raw or " le" in raw:
        return BoolLe(left, right)
    raise UnsupportedIR(f"unsupported comparison predicate: {op.raw}")


def _load_index_from_raw(module: Module, op: Operation) -> tuple[str, IndexExpr]:
    # The v0 textual form is deliberately simple:
    #   %x = tt.load %X[%idx], %mask
    # or:
    #   %x = tt.load %X, %idx, %mask
    bracket = re.search(r"(%[A-Za-z0-9_.$-]+)\s*\[\s*(%[A-Za-z0-9_.$-]+)\s*\]", op.raw)
    if bracket:
        tensor = bracket.group(1).lstrip("%")
        return tensor, recognize_index(module, bracket.group(2))
    if len(op.operands) >= 2:
        tensor = op.operands[0].lstrip("%")
        return tensor, recognize_index(module, op.operands[1])
    raise UnsupportedIR(f"cannot recognize load pointer/index: {op.raw}")


def _store_parts(module: Module, op: Operation) -> tuple[str, IndexExpr, str, str]:
    bracket = re.search(r"(%[A-Za-z0-9_.$-]+)\s*\[\s*(%[A-Za-z0-9_.$-]+)\s*\]", op.raw)
    if bracket:
        output = bracket.group(1).lstrip("%")
        index = recognize_index(module, bracket.group(2))
        remaining = [v for v in op.operands if v not in {bracket.group(1), bracket.group(2)}]
        if len(remaining) < 2:
            raise UnsupportedIR(f"cannot recognize store value/mask: {op.raw}")
        return output, index, remaining[0], remaining[1]
    if len(op.operands) >= 4:
        output = op.operands[0].lstrip("%")
        index = recognize_index(module, op.operands[1])
        return output, index, op.operands[2], op.operands[3]
    raise UnsupportedIR(f"cannot recognize store pointer/index/value/mask: {op.raw}")


def recognize_value(module: Module, value: str, expected_mask: BoolExpr) -> ValueExpr:
    op = _def(module, value)
    if op is None:
        return ValueScalar(value.lstrip("%"))
    if op.op_name in {"arith.constant", "tt.const"}:
        const_match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)", op.raw)
        if not const_match:
            raise UnsupportedIR(f"unsupported value constant: {op.raw}")
        return ValueConst(const_match.group(1))
    if op.op_name in LOAD_OPS:
        tensor, index = _load_index_from_raw(module, op)
        if len(op.operands) >= 3:
            mask_value = op.operands[-1]
            mask = recognize_mask(module, mask_value)
            if mask != expected_mask:
                raise UnsupportedIR(f"load mask differs from store mask: {op.raw}")
        return ValueLoad(tensor, index)
    if op.op_name in FLOAT_OPS:
        if len(op.operands) < 2:
            raise UnsupportedIR(f"binary value op has too few operands: {op.raw}")
        return ValueBinary(
            FLOAT_OPS[op.op_name],
            recognize_value(module, op.operands[0], expected_mask),
            recognize_value(module, op.operands[1], expected_mask),
        )
    if op.op_name in UNARY_OPS:
        if not op.operands:
            raise UnsupportedIR(f"unary value op has no operand: {op.raw}")
        return ValueUnary(UNARY_OPS[op.op_name], recognize_value(module, op.operands[0], expected_mask))
    raise UnsupportedIR(f"unsupported value op in dependency slice: {op.raw}")


def _is_symbol(expr: IndexExpr, name: str) -> bool:
    return isinstance(expr, IndexSymbol) and expr.name == name


def _is_block_extent(expr: IndexExpr, n_symbol: str) -> bool:
    if isinstance(expr, IndexConst):
        return expr.value > 0
    if isinstance(expr, IndexSymbol):
        return expr.name not in {"pid0", "lane", n_symbol}
    return False


def _is_pid_times_block(expr: IndexExpr, n_symbol: str) -> bool:
    expr = simplify_index(expr)
    if not isinstance(expr, IndexMul):
        return False
    return (
        _is_symbol(expr.left, "pid0") and _is_block_extent(expr.right, n_symbol)
    ) or (
        _is_symbol(expr.right, "pid0") and _is_block_extent(expr.left, n_symbol)
    )


def _is_canonical_elementwise_index(expr: IndexExpr, n_symbol: str) -> bool:
    expr = simplify_index(expr)
    if not isinstance(expr, IndexAdd):
        return False
    return (
        _is_pid_times_block(expr.left, n_symbol) and _is_symbol(expr.right, "lane")
    ) or (
        _is_pid_times_block(expr.right, n_symbol) and _is_symbol(expr.left, "lane")
    )


def summarize_kernel(module: Module, n_symbol: str = "N", block_symbol: str = "BLOCK") -> KernelSummary:
    stores = [op for op in module.ops if op.op_name in STORE_OPS]
    if len(stores) != 1:
        raise UnsupportedIR(f"expected exactly one supported store, found {len(stores)}")
    store = stores[0]
    output, store_index, value_id, mask_id = _store_parts(module, store)
    expected_index = simplify_index(store_index)
    if not _is_canonical_elementwise_index(expected_index, n_symbol):
        raise UnsupportedIR(f"store index is not canonical pid*BLOCK+lane: {store.raw}")
    mask = recognize_mask(module, mask_id)
    expected_mask = BoolLt(expected_index, IndexSymbol(n_symbol))
    if mask != expected_mask:
        raise UnsupportedIR(f"store mask is not canonical i < {n_symbol}: {store.raw}")
    value = recognize_value(module, value_id, mask)
    return KernelSummary(
        fragment="elementwise_1d_v0",
        n_symbol=n_symbol,
        block_symbol=block_symbol,
        index=expected_index,
        mask=mask,
        output=output,
        value=value,
    )
