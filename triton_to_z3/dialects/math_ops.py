"""Math dialect handler – covers all operations from the MLIR math dialect.

Operations without a direct Z3 equivalent are modelled as *uninterpreted
functions*, preserving the symbolic structure of the computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import z3

from ..types import FloatType, IntegerType, element_type, is_float_type, make_z3_var, z3_sort

if TYPE_CHECKING:
    from ..interpreter import InterpreterState
    from ..parser import Operation


# ---------------------------------------------------------------------------
# Cache of uninterpreted Z3 functions (one per (name, sort) pair)
# ---------------------------------------------------------------------------

_UNARY_FN_CACHE: dict[tuple[str, z3.SortRef], z3.FuncDeclRef] = {}
_BINARY_FN_CACHE: dict[tuple[str, z3.SortRef], z3.FuncDeclRef] = {}
_TERNARY_FN_CACHE: dict[tuple[str, z3.SortRef], z3.FuncDeclRef] = {}


def _unary_fn(name: str, sort: z3.SortRef) -> z3.FuncDeclRef:
    key = (name, sort)
    if key not in _UNARY_FN_CACHE:
        _UNARY_FN_CACHE[key] = z3.Function(name, sort, sort)
    return _UNARY_FN_CACHE[key]


def _binary_fn(name: str, sort: z3.SortRef) -> z3.FuncDeclRef:
    key = (name, sort)
    if key not in _BINARY_FN_CACHE:
        _BINARY_FN_CACHE[key] = z3.Function(name, sort, sort, sort)
    return _BINARY_FN_CACHE[key]


def _ternary_fn(name: str, sort: z3.SortRef) -> z3.FuncDeclRef:
    key = (name, sort)
    if key not in _TERNARY_FN_CACHE:
        _TERNARY_FN_CACHE[key] = z3.Function(name, sort, sort, sort, sort)
    return _TERNARY_FN_CACHE[key]


# ---------------------------------------------------------------------------
# Unary / binary / ternary FP helpers via uninterpreted functions
# ---------------------------------------------------------------------------

def _apply_unary_uif(
    op: "Operation",
    state: "InterpreterState",
    fn_name: str,
) -> None:
    if not op.operands or not op.results:
        return
    arg = state.get_fp(op.operands[0])
    sort = arg.sort() if z3.is_fp(arg) else z3.Float32()
    result_type = op.result_types[0] if op.result_types else None
    state.set(op.results[0], _unary_fn(fn_name, sort)(arg), result_type)


def _apply_binary_uif(
    op: "Operation",
    state: "InterpreterState",
    fn_name: str,
) -> None:
    if len(op.operands) < 2 or not op.results:
        return
    lhs = state.get_fp(op.operands[0])
    rhs = state.get_fp(op.operands[1])
    sort = lhs.sort() if z3.is_fp(lhs) else z3.Float32()
    result_type = op.result_types[0] if op.result_types else None
    state.set(op.results[0], _binary_fn(fn_name, sort)(lhs, rhs), result_type)


def _apply_ternary_uif(
    op: "Operation",
    state: "InterpreterState",
    fn_name: str,
) -> None:
    if len(op.operands) < 3 or not op.results:
        return
    a = state.get_fp(op.operands[0])
    b = state.get_fp(op.operands[1])
    c = state.get_fp(op.operands[2])
    sort = a.sort() if z3.is_fp(a) else z3.Float32()
    result_type = op.result_types[0] if op.result_types else None
    state.set(op.results[0], _ternary_fn(fn_name, sort)(a, b, c), result_type)


# ---------------------------------------------------------------------------
# Integer unary helpers
# ---------------------------------------------------------------------------

def _apply_int_unary_uif(
    op: "Operation",
    state: "InterpreterState",
    fn_name: str,
) -> None:
    if not op.operands or not op.results:
        return
    arg = state.get_bv(op.operands[0])
    sort = arg.sort() if z3.is_bv(arg) else z3.BitVecSort(32)
    fn = z3.Function(fn_name, sort, sort)
    result_type = op.result_types[0] if op.result_types else None
    state.set(op.results[0], fn(arg), result_type)


# ---------------------------------------------------------------------------
# Classification ops (return bool)
# ---------------------------------------------------------------------------

def _classification(
    op: "Operation",
    state: "InterpreterState",
    z3_fn: object,
) -> None:
    if not op.operands or not op.results:
        return
    arg = state.get_fp(op.operands[0])
    state.set(op.results[0], z3_fn(arg))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def handle(op: "Operation", state: "InterpreterState") -> None:
    name = op.op.removeprefix("math.")

    match name:
        # --- Absolute value ----------------------------------------------
        case "absf":
            _apply_unary(op, state, lambda x: z3.fpAbs(x))
        case "absi":
            if op.operands and op.results:
                arg = state.get_bv(op.operands[0])
                if z3.is_bv(arg):
                    neg = -arg
                    state.set(op.results[0], z3.If(arg < 0, neg, arg))
                else:
                    _apply_int_unary_uif(op, state, "absi")

        # --- Trigonometric -----------------------------------------------
        case "cos":
            _apply_unary_uif(op, state, "cos")
        case "sin":
            _apply_unary_uif(op, state, "sin")
        case "tan":
            _apply_unary_uif(op, state, "tan")
        case "cosh":
            _apply_unary_uif(op, state, "cosh")
        case "sinh":
            _apply_unary_uif(op, state, "sinh")
        case "tanh":
            _apply_unary_uif(op, state, "tanh")

        # --- Inverse trigonometric ---------------------------------------
        case "acos":
            _apply_unary_uif(op, state, "acos")
        case "asin":
            _apply_unary_uif(op, state, "asin")
        case "atan":
            _apply_unary_uif(op, state, "atan")
        case "atan2":
            _apply_binary_uif(op, state, "atan2")
        case "acosh":
            _apply_unary_uif(op, state, "acosh")
        case "asinh":
            _apply_unary_uif(op, state, "asinh")
        case "atanh":
            _apply_unary_uif(op, state, "atanh")

        # --- sincos (two results) ----------------------------------------
        case "sincos":
            if op.operands and len(op.results) >= 2:
                arg = state.get_fp(op.operands[0])
                sort = arg.sort() if z3.is_fp(arg) else z3.Float32()
                state.set(op.results[0], _unary_fn("sin", sort)(arg))
                state.set(op.results[1], _unary_fn("cos", sort)(arg))

        # --- Exponential / logarithmic -----------------------------------
        case "exp":
            _apply_unary_uif(op, state, "exp")
        case "exp2":
            _apply_unary_uif(op, state, "exp2")
        case "expm1":
            _apply_unary_uif(op, state, "expm1")
        case "log":
            _apply_unary_uif(op, state, "log")
        case "log2":
            _apply_unary_uif(op, state, "log2")
        case "log10":
            _apply_unary_uif(op, state, "log10")
        case "log1p":
            _apply_unary_uif(op, state, "log1p")

        # --- Power / roots -----------------------------------------------
        case "powf":
            _apply_binary_uif(op, state, "powf")
        case "fpowi":
            # float ** int – model as uninterpreted
            _apply_binary_uif(op, state, "fpowi")
        case "ipowi":
            _apply_int_unary_uif(op, state, "ipowi")  # approximation
        case "sqrt":
            _apply_unary(op, state, lambda x: z3.fpSqrt(z3.RNE(), x))
        case "rsqrt":
            _apply_unary_uif(op, state, "rsqrt")
        case "cbrt":
            _apply_unary_uif(op, state, "cbrt")

        # --- Rounding ----------------------------------------------------
        case "ceil":
            _apply_unary(op, state, lambda x: z3.fpRoundToIntegral(z3.RTP(), x))
        case "floor":
            _apply_unary(op, state, lambda x: z3.fpRoundToIntegral(z3.RTN(), x))
        case "round":
            _apply_unary(op, state, lambda x: z3.fpRoundToIntegral(z3.RNA(), x))
        case "roundeven":
            _apply_unary(op, state, lambda x: z3.fpRoundToIntegral(z3.RNE(), x))
        case "trunc":
            _apply_unary(op, state, lambda x: z3.fpRoundToIntegral(z3.RTZ(), x))

        # --- Clamping ----------------------------------------------------
        case "clampf":
            _apply_ternary_uif(op, state, "clampf")

        # --- Copy sign ---------------------------------------------------
        case "copysign":
            if len(op.operands) >= 2 and op.results:
                mag = state.get_fp(op.operands[0])
                sgn = state.get_fp(op.operands[1])
                # copysign(mag, sgn): magnitude of mag, sign of sgn
                abs_mag = z3.fpAbs(mag)
                neg_mag = z3.fpNeg(abs_mag)
                is_neg = z3.fpIsNegative(sgn)
                result_type = op.result_types[0] if op.result_types else None
                state.set(op.results[0], z3.If(is_neg, neg_mag, abs_mag), result_type)

        # --- FMA (fused multiply-add) ------------------------------------
        case "fma":
            if len(op.operands) >= 3 and op.results:
                a = state.get_fp(op.operands[0])
                b = state.get_fp(op.operands[1])
                c = state.get_fp(op.operands[2])
                result_type = op.result_types[0] if op.result_types else None
                state.set(
                    op.results[0],
                    z3.fpFMA(z3.RNE(), a, b, c),
                    result_type,
                )

        # --- Error function ----------------------------------------------
        case "erf":
            _apply_unary_uif(op, state, "erf")
        case "erfc":
            _apply_unary_uif(op, state, "erfc")

        # --- Bit-counting (integer) --------------------------------------
        case "ctlz":
            _apply_int_unary_uif(op, state, "ctlz")
        case "cttz":
            _apply_int_unary_uif(op, state, "cttz")
        case "ctpop":
            _apply_int_unary_uif(op, state, "ctpop")

        # --- Classification (return i1 / bool) ---------------------------
        case "isnan":
            _classification(op, state, z3.fpIsNaN)
        case "isinf":
            _classification(op, state, z3.fpIsInf)
        case "isfinite":
            _classification(op, state, lambda x: z3.Not(z3.Or(z3.fpIsNaN(x), z3.fpIsInf(x))))
        case "isnormal":
            _classification(op, state, z3.fpIsNormal)

        # --- Fallback ----------------------------------------------------
        case _:
            for r in op.results:
                state.set_unknown(r, op.op)


# ---------------------------------------------------------------------------
# Helpers that use native Z3 FP operations (not uninterpreted)
# ---------------------------------------------------------------------------

def _apply_unary(
    op: "Operation",
    state: "InterpreterState",
    fn: object,
) -> None:
    if not op.operands or not op.results:
        return
    arg = state.get_fp(op.operands[0])
    result_type = op.result_types[0] if op.result_types else None
    state.set(op.results[0], fn(arg), result_type)
