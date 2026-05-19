"""Arith dialect handler – covers all operations from the MLIR arith dialect."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import z3

from ..types import (
    FloatType,
    IntegerType,
    MLIRType,
    BFloat16Type,
    IndexType,
    PointerType,
    TensorType,
    element_type,
    fp_sort_for_width,
    is_float_type,
    make_z3_var,
    parse_type,
    z3_sort,
)

if TYPE_CHECKING:
    from ..interpreter import InterpreterState
    from ..parser import Operation


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def handle(op: "Operation", state: "InterpreterState") -> None:
    name = op.op.removeprefix("arith.")
    match name:
        # --- constants ---------------------------------------------------
        case "constant":
            _handle_constant(op, state)

        # --- floating-point binary arithmetic ----------------------------
        case "addf":
            _binop_fp(op, state, lambda l, r: z3.fpAdd(z3.RNE(), l, r))
        case "subf":
            _binop_fp(op, state, lambda l, r: z3.fpSub(z3.RNE(), l, r))
        case "mulf":
            _binop_fp(op, state, lambda l, r: z3.fpMul(z3.RNE(), l, r))
        case "divf":
            _binop_fp(op, state, lambda l, r: z3.fpDiv(z3.RNE(), l, r))
        case "remf":
            _binop_fp(op, state, lambda l, r: z3.fpRem(l, r))

        # --- floating-point unary ----------------------------------------
        case "negf":
            _unop_fp(op, state, lambda x: z3.fpNeg(x))

        # --- floating-point min / max ------------------------------------
        case "maximumf" | "maxnumf":
            _binop_fp(op, state, lambda l, r: z3.If(z3.fpGT(l, r), l, r))
        case "minimumf" | "minnumf":
            _binop_fp(op, state, lambda l, r: z3.If(z3.fpLT(l, r), l, r))

        # --- integer binary arithmetic -----------------------------------
        case "addi":
            _binop_bv(op, state, lambda l, r: l + r)
        case "subi":
            _binop_bv(op, state, lambda l, r: l - r)
        case "muli":
            _binop_bv(op, state, lambda l, r: l * r)
        case "divsi":
            _binop_bv(op, state, lambda l, r: l / r)
        case "divui":
            _binop_bv(op, state, lambda l, r: z3.UDiv(l, r))
        case "ceildivsi":
            _binop_bv(op, state, _ceildivsi)
        case "ceildivui":
            _binop_bv(op, state, lambda l, r: z3.UDiv(l + r - z3.BitVecVal(1, l.size()), r))
        case "floordivsi":
            _binop_bv(op, state, _floordivsi)
        case "remsi":
            _binop_bv(op, state, lambda l, r: z3.SRem(l, r))
        case "remui":
            _binop_bv(op, state, lambda l, r: z3.URem(l, r))

        # --- integer extended arithmetic ---------------------------------
        case "addui_extended":
            _extended_add(op, state)
        case "subui_extended":
            _extended_sub(op, state)
        case "mulsi_extended" | "mului_extended":
            _extended_mul(op, state, signed=(name == "mulsi_extended"))

        # --- integer min / max -------------------------------------------
        case "maxsi":
            _binop_bv(op, state, lambda l, r: z3.If(l > r, l, r))
        case "minsi":
            _binop_bv(op, state, lambda l, r: z3.If(l < r, l, r))
        case "maxui":
            _binop_bv(op, state, lambda l, r: z3.If(z3.UGT(l, r), l, r))
        case "minui":
            _binop_bv(op, state, lambda l, r: z3.If(z3.ULT(l, r), l, r))

        # --- bitwise operations ------------------------------------------
        case "andi":
            _binop_bv(op, state, lambda l, r: l & r)
        case "ori":
            _binop_bv(op, state, lambda l, r: l | r)
        case "xori":
            _binop_bv(op, state, lambda l, r: l ^ r)

        # --- shift operations --------------------------------------------
        case "shli":
            _binop_bv(op, state, lambda l, r: l << r)
        case "shrsi":
            _binop_bv(op, state, lambda l, r: l >> r)
        case "shrui":
            _binop_bv(op, state, lambda l, r: z3.LShR(l, r))

        # --- comparisons -------------------------------------------------
        case "cmpi":
            _handle_cmpi(op, state)
        case "cmpf":
            _handle_cmpf(op, state)

        # --- select ------------------------------------------------------
        case "select":
            _handle_select(op, state)

        # --- float → float casts -----------------------------------------
        case "extf":
            _handle_fp_cast(op, state)
        case "truncf":
            _handle_fp_cast(op, state)
        case "convertf":
            _handle_fp_cast(op, state)

        # --- int → int casts ---------------------------------------------
        case "extsi":
            _handle_int_ext(op, state, signed=True)
        case "extui":
            _handle_int_ext(op, state, signed=False)
        case "trunci":
            _handle_int_trunc(op, state)

        # --- int ↔ float casts -------------------------------------------
        case "sitofp":
            _handle_int_to_fp(op, state, signed=True)
        case "uitofp":
            _handle_int_to_fp(op, state, signed=False)
        case "fptosi":
            _handle_fp_to_int(op, state, signed=True)
        case "fptoui":
            _handle_fp_to_int(op, state, signed=False)

        # --- index casts -------------------------------------------------
        case "index_cast" | "index_castui":
            _handle_index_cast(op, state)

        # --- bitcast -----------------------------------------------------
        case "bitcast":
            _handle_bitcast(op, state)

        # --- fallback: unknown arith op ----------------------------------
        case _:
            for r in op.results:
                state.set_unknown(r, op.op)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEX_FLOAT_RE = re.compile(r"0x[0-9A-Fa-f]+")


def _handle_constant(op: "Operation", state: "InterpreterState") -> None:
    if not op.results:
        return
    res = op.results[0]
    val_str = op.attributes.get("value", "").strip()
    result_type = op.result_types[0] if op.result_types else None
    etype = element_type(result_type) if result_type else None

    # dense<...> tensor constants
    if val_str.startswith("dense"):
        inner = re.search(r"dense<([^>]+)>", val_str)
        if inner:
            raw = inner.group(1).strip()
            state.set(res, _parse_scalar_constant(raw, etype), result_type)
        else:
            state.set(res, make_z3_var(f"const_{res}", result_type or FloatType(32)), result_type)
        return

    # Scalar constant
    state.set(res, _parse_scalar_constant(val_str, etype), result_type)


def _parse_scalar_constant(raw: str, etype: MLIRType | None) -> z3.ExprRef:
    """Parse a scalar constant value string into a Z3 expression."""
    raw = raw.strip()

    # Hex float patterns (IEEE 754 bit patterns)
    if _HEX_FLOAT_RE.fullmatch(raw):
        int_val = int(raw, 16)
        match int_val:
            case 0xFF800000:
                return z3.fpMinusInfinity(z3.Float32())
            case 0x7F800000:
                return z3.fpPlusInfinity(z3.Float32())
            case 0xFFF0000000000000:
                return z3.fpMinusInfinity(z3.Float64())
            case 0x7FF0000000000000:
                return z3.fpPlusInfinity(z3.Float64())
            case _:
                # Treat as float bit pattern
                sort = etype.to_z3_sort() if etype and is_float_type(etype) else z3.Float32()
                return z3.fpBVToFP(z3.BitVecVal(int_val, sort.ebits() + sort.sbits()), sort)

    # Boolean
    if raw == "true":
        return z3.BoolVal(True)
    if raw == "false":
        return z3.BoolVal(False)

    # Try float
    if etype and is_float_type(etype):
        try:
            sort = etype.to_z3_sort()
            return z3.FPVal(float(raw.split(":")[0].strip()), sort)
        except (ValueError, z3.Z3Exception):
            pass

    # Try integer
    try:
        int_val = int(raw.split(":")[0].strip())
        if etype and isinstance(element_type(etype) if isinstance(etype, TensorType) else etype, IntegerType):
            w = (element_type(etype) if isinstance(etype, TensorType) else etype).width
            if w == 1:
                return z3.BoolVal(bool(int_val))
            return z3.BitVecVal(int_val, w)
        if etype and isinstance(element_type(etype) if isinstance(etype, TensorType) else etype, IndexType):
            return z3.BitVecVal(int_val, 64)
        return z3.BitVecVal(int_val, 32)
    except (ValueError, TypeError):
        pass

    # Try float as fallback
    try:
        return z3.FPVal(float(raw.split(":")[0].strip()), z3.Float32())
    except (ValueError, z3.Z3Exception):
        pass

    # Last resort: symbolic constant
    sort = etype.to_z3_sort() if etype else z3.Float32()
    if isinstance(sort, z3.FPSortRef):
        return z3.FP(f"const_{raw}", sort)
    return z3.BitVec(f"const_{raw}", sort.size() if hasattr(sort, "size") else 32)


# ---------------------------------------------------------------------------
# Binary / unary helpers
# ---------------------------------------------------------------------------

def _binop_fp(
    op: "Operation",
    state: "InterpreterState",
    fn: object,
) -> None:
    if len(op.operands) < 2 or not op.results:
        return
    lhs = state.get_fp(op.operands[0])
    rhs = state.get_fp(op.operands[1])
    result_type = op.result_types[0] if op.result_types else None
    state.set(op.results[0], fn(lhs, rhs), result_type)


def _unop_fp(
    op: "Operation",
    state: "InterpreterState",
    fn: object,
) -> None:
    if not op.operands or not op.results:
        return
    arg = state.get_fp(op.operands[0])
    result_type = op.result_types[0] if op.result_types else None
    state.set(op.results[0], fn(arg), result_type)


def _binop_bv(
    op: "Operation",
    state: "InterpreterState",
    fn: object,
) -> None:
    if len(op.operands) < 2 or not op.results:
        return
    lhs = state.get_bv(op.operands[0])
    rhs = state.get_bv(op.operands[1])
    # Match bit widths if different
    lhs, rhs = _match_bv_widths(lhs, rhs)
    result_type = op.result_types[0] if op.result_types else None
    state.set(op.results[0], fn(lhs, rhs), result_type)


def _match_bv_widths(a: z3.ExprRef, b: z3.ExprRef) -> tuple[z3.ExprRef, z3.ExprRef]:
    """Zero-extend the narrower bitvector to match the wider one."""
    if not (z3.is_bv(a) and z3.is_bv(b)):
        return a, b
    wa, wb = a.size(), b.size()
    if wa < wb:
        a = z3.ZeroExt(wb - wa, a)
    elif wb < wa:
        b = z3.ZeroExt(wa - wb, b)
    return a, b


# ---------------------------------------------------------------------------
# Integer division helpers
# ---------------------------------------------------------------------------

def _ceildivsi(l: z3.ExprRef, r: z3.ExprRef) -> z3.ExprRef:
    one = z3.BitVecVal(1, l.size())
    # ceildiv(a, b) when both positive: (a + b - 1) / b
    return (l + r - one) / r


def _floordivsi(l: z3.ExprRef, r: z3.ExprRef) -> z3.ExprRef:
    return l / r  # Z3 signed division truncates toward zero; close enough for symbolic analysis


# ---------------------------------------------------------------------------
# Extended arithmetic
# ---------------------------------------------------------------------------

def _extended_add(op: "Operation", state: "InterpreterState") -> None:
    if len(op.operands) < 2 or len(op.results) < 2:
        return
    lhs = state.get_bv(op.operands[0])
    rhs = state.get_bv(op.operands[1])
    lhs, rhs = _match_bv_widths(lhs, rhs)
    w = lhs.size()
    ext_l = z3.ZeroExt(1, lhs)
    ext_r = z3.ZeroExt(1, rhs)
    full = ext_l + ext_r
    state.set(op.results[0], z3.Extract(w - 1, 0, full))
    state.set(op.results[1], z3.Extract(w, w, full) == z3.BitVecVal(1, 1))


def _extended_sub(op: "Operation", state: "InterpreterState") -> None:
    if len(op.operands) < 2 or len(op.results) < 2:
        return
    lhs = state.get_bv(op.operands[0])
    rhs = state.get_bv(op.operands[1])
    lhs, rhs = _match_bv_widths(lhs, rhs)
    w = lhs.size()
    state.set(op.results[0], lhs - rhs)
    state.set(op.results[1], z3.UGT(rhs, lhs))  # borrow if rhs > lhs (unsigned)


def _extended_mul(op: "Operation", state: "InterpreterState", *, signed: bool) -> None:
    if len(op.operands) < 2 or len(op.results) < 2:
        return
    lhs = state.get_bv(op.operands[0])
    rhs = state.get_bv(op.operands[1])
    lhs, rhs = _match_bv_widths(lhs, rhs)
    w = lhs.size()
    ext = z3.SignExt if signed else z3.ZeroExt
    full = ext(w, lhs) * ext(w, rhs)
    state.set(op.results[0], z3.Extract(w - 1, 0, full))
    state.set(op.results[1], z3.Extract(2 * w - 1, w, full))


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def _handle_cmpi(op: "Operation", state: "InterpreterState") -> None:
    if len(op.operands) < 2 or not op.results:
        return
    pred = op.attributes.get("predicate", "eq")
    lhs = state.get_bv(op.operands[0])
    rhs = state.get_bv(op.operands[1])
    lhs, rhs = _match_bv_widths(lhs, rhs)

    match pred:
        case "eq":  result = lhs == rhs
        case "ne":  result = lhs != rhs
        case "slt": result = lhs < rhs
        case "sle": result = lhs <= rhs
        case "sgt": result = lhs > rhs
        case "sge": result = lhs >= rhs
        case "ult": result = z3.ULT(lhs, rhs)
        case "ule": result = z3.ULE(lhs, rhs)
        case "ugt": result = z3.UGT(lhs, rhs)
        case "uge": result = z3.UGE(lhs, rhs)
        case _:     result = lhs == rhs

    state.set(op.results[0], result)


def _handle_cmpf(op: "Operation", state: "InterpreterState") -> None:
    if len(op.operands) < 2 or not op.results:
        return
    pred = op.attributes.get("predicate", "oeq")
    lhs = state.get_fp(op.operands[0])
    rhs = state.get_fp(op.operands[1])

    match pred:
        case "oeq":         result = z3.fpEQ(lhs, rhs)
        case "ogt":         result = z3.fpGT(lhs, rhs)
        case "oge":         result = z3.fpGEQ(lhs, rhs)
        case "olt":         result = z3.fpLT(lhs, rhs)
        case "ole":         result = z3.fpLEQ(lhs, rhs)
        case "one":         result = z3.Or(z3.fpLT(lhs, rhs), z3.fpGT(lhs, rhs))
        case "ord":         result = z3.Not(z3.Or(z3.fpIsNaN(lhs), z3.fpIsNaN(rhs)))
        case "ueq":         result = z3.Not(z3.Or(z3.fpLT(lhs, rhs), z3.fpGT(lhs, rhs)))
        case "ugt":         result = z3.Not(z3.fpLEQ(lhs, rhs))
        case "uge":         result = z3.Not(z3.fpLT(lhs, rhs))
        case "ult":         result = z3.Not(z3.fpGEQ(lhs, rhs))
        case "ule":         result = z3.Not(z3.fpGT(lhs, rhs))
        case "une":         result = z3.Not(z3.fpEQ(lhs, rhs))
        case "uno":         result = z3.Or(z3.fpIsNaN(lhs), z3.fpIsNaN(rhs))
        case "always_true": result = z3.BoolVal(True)
        case "always_false":result = z3.BoolVal(False)
        case _:             result = z3.fpEQ(lhs, rhs)

    state.set(op.results[0], result)


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------

def _handle_select(op: "Operation", state: "InterpreterState") -> None:
    if len(op.operands) < 3 or not op.results:
        return
    cond = state.get_bool(op.operands[0])
    true_val = state.get(op.operands[1])
    false_val = state.get(op.operands[2])
    result_type = op.result_types[-1] if op.result_types else None
    state.set(op.results[0], z3.If(cond, true_val, false_val), result_type)


# ---------------------------------------------------------------------------
# Type casts
# ---------------------------------------------------------------------------

def _result_type_of(op: "Operation") -> MLIRType | None:
    if op.result_types:
        return op.result_types[-1]
    return None


def _handle_fp_cast(op: "Operation", state: "InterpreterState") -> None:
    """extf / truncf / convertf → fpToFP with target sort."""
    if not op.operands or not op.results:
        return
    src = state.get_fp(op.operands[0])
    target = _result_type_of(op)
    target_sort = z3_sort(target) if target else z3.Float32()
    if isinstance(target_sort, z3.FPSortRef):
        state.set(op.results[0], z3.fpToFP(z3.RNE(), src, target_sort), target)
    else:
        state.set(op.results[0], src, target)


def _handle_int_ext(op: "Operation", state: "InterpreterState", *, signed: bool) -> None:
    if not op.operands or not op.results:
        return
    src = state.get_bv(op.operands[0])
    target = _result_type_of(op)
    target_width = 64
    if target:
        from ..types import bv_width
        target_width = bv_width(target)
    if z3.is_bv(src):
        ext_bits = max(0, target_width - src.size())
        ext_fn = z3.SignExt if signed else z3.ZeroExt
        state.set(op.results[0], ext_fn(ext_bits, src) if ext_bits > 0 else src, target)
    else:
        state.set(op.results[0], src, target)


def _handle_int_trunc(op: "Operation", state: "InterpreterState") -> None:
    if not op.operands or not op.results:
        return
    src = state.get_bv(op.operands[0])
    target = _result_type_of(op)
    target_width = 32
    if target:
        from ..types import bv_width
        target_width = bv_width(target)
    if z3.is_bv(src):
        state.set(op.results[0], z3.Extract(target_width - 1, 0, src), target)
    else:
        state.set(op.results[0], src, target)


def _handle_int_to_fp(op: "Operation", state: "InterpreterState", *, signed: bool) -> None:
    if not op.operands or not op.results:
        return
    src = state.get_bv(op.operands[0])
    target = _result_type_of(op)
    target_sort = z3_sort(target) if target else z3.Float32()
    if z3.is_bv(src) and isinstance(target_sort, z3.FPSortRef):
        if signed:
            state.set(op.results[0], z3.fpSignedToFP(z3.RNE(), src, target_sort), target)
        else:
            state.set(op.results[0], z3.fpUnsignedToFP(z3.RNE(), src, target_sort), target)
    else:
        state.set(op.results[0], make_z3_var(op.results[0], target or FloatType(32)), target)


def _handle_fp_to_int(op: "Operation", state: "InterpreterState", *, signed: bool) -> None:
    if not op.operands or not op.results:
        return
    src = state.get_fp(op.operands[0])
    target = _result_type_of(op)
    target_width = 32
    if target:
        from ..types import bv_width
        target_width = bv_width(target)
    if z3.is_fp(src):
        if signed:
            state.set(op.results[0], z3.fpToSBV(z3.RTZ(), src, z3.BitVecSort(target_width)), target)
        else:
            state.set(op.results[0], z3.fpToUBV(z3.RTZ(), src, z3.BitVecSort(target_width)), target)
    else:
        state.set(op.results[0], make_z3_var(op.results[0], target or IntegerType(32)), target)


def _handle_index_cast(op: "Operation", state: "InterpreterState") -> None:
    if not op.operands or not op.results:
        return
    src = state.get_bv(op.operands[0])
    target = _result_type_of(op)
    target_width = 64
    if target:
        from ..types import bv_width
        target_width = bv_width(target)
    if z3.is_bv(src):
        sw = src.size()
        if sw < target_width:
            state.set(op.results[0], z3.SignExt(target_width - sw, src), target)
        elif sw > target_width:
            state.set(op.results[0], z3.Extract(target_width - 1, 0, src), target)
        else:
            state.set(op.results[0], src, target)
    else:
        state.set(op.results[0], src, target)


def _handle_bitcast(op: "Operation", state: "InterpreterState") -> None:
    """Bitcast – reinterpret bits between types of the same width."""
    if not op.operands or not op.results:
        return
    src = state.get(op.operands[0])
    target = _result_type_of(op)
    target_sort = z3_sort(target) if target else None

    if target_sort is not None and isinstance(target_sort, z3.FPSortRef) and z3.is_bv(src):
        state.set(op.results[0], z3.fpBVToFP(src, target_sort), target)
    elif target_sort is not None and z3.is_fp(src) and not isinstance(target_sort, z3.FPSortRef):
        state.set(op.results[0], z3.fpToIEEEBV(src), target)
    else:
        state.set(op.results[0], src, target)
