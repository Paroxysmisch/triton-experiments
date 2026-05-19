"""Triton (``tt.*``) dialect handler.

Covers load/store, pointer arithmetic, tensor shape manipulation,
reductions, extern elementwise calls, dot products, and program-id queries.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import z3

from ..types import (
    FloatType,
    IntegerType,
    IndexType,
    PointerType,
    TensorType,
    element_type,
    is_float_type,
    make_z3_var,
)

if TYPE_CHECKING:
    from ..interpreter import InterpreterState
    from ..parser import Operation


# ---------------------------------------------------------------------------
# Well-known extern symbols → Z3 uninterpreted-function names
# ---------------------------------------------------------------------------

_EXTERN_SYMBOL_MAP: dict[str, str] = {
    "__nv_expf": "exp",
    "__nv_exp": "exp",
    "__nv_exp2f": "exp2",
    "__nv_logf": "log",
    "__nv_log": "log",
    "__nv_log2f": "log2",
    "__nv_log10f": "log10",
    "__nv_sqrtf": "sqrt",
    "__nv_rsqrtf": "rsqrt",
    "__nv_sinf": "sin",
    "__nv_cosf": "cos",
    "__nv_tanf": "tan",
    "__nv_tanhf": "tanh",
    "__nv_erff": "erf",
    "__nv_fabsf": "absf",
    "__nv_fmaf": "fma",
    "__nv_powf": "powf",
    "__nv_ceilf": "ceil",
    "__nv_floorf": "floor",
    "__nv_fmodf": "fmod",
    "__nv_fmaxf": "fmax",
    "__nv_fminf": "fmin",
    "__nv_copysignf": "copysign",
    "__nv_fast_expf": "exp",
    "__nv_fast_logf": "log",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def handle(op: "Operation", state: "InterpreterState") -> None:
    name = op.op.removeprefix("tt.")
    match name:
        # --- Memory operations -------------------------------------------
        case "load":
            _handle_load(op, state)
        case "store":
            _handle_store(op, state)

        # --- Pointer arithmetic ------------------------------------------
        case "addptr":
            _handle_addptr(op, state)

        # --- Shape / broadcast / splat -----------------------------------
        case "splat":
            _handle_alias(op, state)
        case "broadcast":
            _handle_alias(op, state)
        case "expand_dims":
            _handle_alias(op, state)
        case "reshape":
            _handle_alias(op, state)
        case "cat":
            _handle_alias(op, state)
        case "trans":
            _handle_alias(op, state)
        case "make_tensor_ptr":
            _handle_make_tensor_ptr(op, state)
        case "advance":
            _handle_alias(op, state)

        # --- Range -------------------------------------------------------
        case "make_range":
            _handle_make_range(op, state)

        # --- Program ID / grid info --------------------------------------
        case "get_program_id":
            _handle_get_program_id(op, state)
        case "get_num_programs":
            _handle_get_num_programs(op, state)

        # --- Reductions --------------------------------------------------
        case "reduce":
            _handle_reduce(op, state)
        case "reduce.return":
            _handle_reduce_return(op, state)
        case "scan":
            _handle_scan(op, state)

        # --- Extern elementwise ------------------------------------------
        case "extern_elementwise":
            _handle_extern_elementwise(op, state)

        # --- Dot product (matmul) ----------------------------------------
        case "dot":
            _handle_dot(op, state)

        # --- Atomic operations -------------------------------------------
        case "atomic_cas":
            _handle_atomic(op, state, "atomic_cas")
        case "atomic_rmw":
            _handle_atomic(op, state, "atomic_rmw")

        # --- Conversion / cast -------------------------------------------
        case "bitcast":
            _handle_alias(op, state)
        case "fp_to_fp" | "int_to_fp" | "fp_to_int":
            _handle_tt_cast(op, state)

        # --- Control flow ------------------------------------------------
        case "func":
            pass  # Handled at top-level parse
        case "return":
            pass
        case "call":
            _handle_call(op, state)
        case "print":
            pass

        # --- Fallback ----------------------------------------------------
        case _:
            for r in op.results:
                state.set_unknown(r, op.op)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _handle_load(op: "Operation", state: "InterpreterState") -> None:
    """tt.load produces a fresh terminal symbol (the loaded value is unknown)."""
    if not op.results:
        return
    res = op.results[0]

    # Determine element type from the pointer operand's type or result type
    result_type = op.result_types[0] if op.result_types else None
    if result_type:
        etype = element_type(result_type)
    else:
        etype = FloatType(32)

    sym = make_z3_var(f"load_{res}", etype)
    state.set(res, sym, result_type)
    state.terminals.add(res)


def _handle_store(op: "Operation", state: "InterpreterState") -> None:
    """tt.store is a side-effect; record the stored value for later extraction."""
    # tt.store %ptr, %val[, %mask] : type
    if len(op.operands) >= 2:
        ptr_name = op.operands[0]
        val_name = op.operands[1]
        state.stores.append((ptr_name, val_name))


# ---------------------------------------------------------------------------
# Pointer arithmetic
# ---------------------------------------------------------------------------

def _handle_addptr(op: "Operation", state: "InterpreterState") -> None:
    """tt.addptr %ptr, %offset → ptr + offset."""
    if len(op.operands) < 2 or not op.results:
        return
    ptr = state.get(op.operands[0])
    offset = state.get(op.operands[1])

    # Both should be bitvectors for pointer arithmetic
    if z3.is_bv(ptr) and z3.is_bv(offset):
        pw, ow = ptr.size(), offset.size()
        if ow < pw:
            offset = z3.SignExt(pw - ow, offset)
        elif pw < ow:
            ptr = z3.ZeroExt(ow - pw, ptr)
        result = ptr + offset
    elif z3.is_bv(ptr):
        result = ptr  # Can't add non-bv offset; keep ptr
    else:
        # Both may be symbolic; just keep the pointer value
        result = ptr

    result_type = op.result_types[0] if op.result_types else None
    state.set(op.results[0], result, result_type)


# ---------------------------------------------------------------------------
# Shape / broadcast / splat  (aliases in symbolic execution)
# ---------------------------------------------------------------------------

def _handle_alias(op: "Operation", state: "InterpreterState") -> None:
    """Splat/broadcast/expand_dims/reshape/trans – the symbolic value is unchanged."""
    if not op.operands or not op.results:
        return
    src = state.get(op.operands[0])
    result_type = op.result_types[0] if op.result_types else None
    state.set(op.results[0], src, result_type)


# ---------------------------------------------------------------------------
# make_range
# ---------------------------------------------------------------------------

def _handle_make_range(op: "Operation", state: "InterpreterState") -> None:
    """tt.make_range {start, end} → fresh symbolic integer."""
    if not op.results:
        return
    res = op.results[0]
    result_type = op.result_types[0] if op.result_types else IntegerType(32)
    sym = make_z3_var(res, result_type)
    state.set(res, sym, result_type)
    state.terminals.add(res)


# ---------------------------------------------------------------------------
# Program ID / grid
# ---------------------------------------------------------------------------

def _handle_get_program_id(op: "Operation", state: "InterpreterState") -> None:
    if not op.results:
        return
    res = op.results[0]
    # Determine axis from operands text or attribute
    axis = op.attributes.get("axis", "x")
    sym = z3.BitVec(f"program_id_{axis}", 32)
    state.set(res, sym, IntegerType(32))
    state.terminals.add(res)


def _handle_get_num_programs(op: "Operation", state: "InterpreterState") -> None:
    if not op.results:
        return
    res = op.results[0]
    axis = op.attributes.get("axis", "x")
    sym = z3.BitVec(f"num_programs_{axis}", 32)
    state.set(res, sym, IntegerType(32))
    state.terminals.add(res)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def _handle_reduce(op: "Operation", state: "InterpreterState") -> None:
    """tt.reduce – the result is a terminal; the combiner region is recorded as metadata."""
    if not op.results:
        return

    from ..interpreter import interpret_ops

    res = op.results[0]
    result_type = op.result_types[0] if op.result_types else None

    # Determine the reduction combiner by inspecting the region body
    combiner_name = "reduce"
    if op.regions:
        region = op.regions[0]
        for body_op in region.body:
            if body_op.op != "tt.reduce.return":
                combiner_name = body_op.op.split(".")[-1]  # e.g. "addf", "maxnumf"
                break

    sym = make_z3_var(f"{combiner_name}_{res}", result_type or FloatType(32))
    state.set(res, sym, result_type)
    state.terminals.add(res)


def _handle_reduce_return(op: "Operation", state: "InterpreterState") -> None:
    """Inside a reduce region – the return value feeds back to the combiner."""
    # In our symbolic model, reductions produce a terminal.
    # The reduce.return is consumed by _handle_reduce above.
    pass


def _handle_scan(op: "Operation", state: "InterpreterState") -> None:
    """tt.scan (prefix scan) – similar to reduce, result is a terminal."""
    if not op.results:
        return
    res = op.results[0]
    result_type = op.result_types[0] if op.result_types else None
    sym = make_z3_var(f"scan_{res}", result_type or FloatType(32))
    state.set(res, sym, result_type)
    state.terminals.add(res)


# ---------------------------------------------------------------------------
# Extern elementwise
# ---------------------------------------------------------------------------

def _handle_extern_elementwise(op: "Operation", state: "InterpreterState") -> None:
    """tt.extern_elementwise – map known symbols to uninterpreted functions."""
    if not op.results:
        return
    res = op.results[0]
    sym_name = op.attributes.get("symbol", "").strip('"')
    fn_name = _EXTERN_SYMBOL_MAP.get(sym_name, sym_name or f"extern_{res}")

    result_type = op.result_types[0] if op.result_types else None
    etype = element_type(result_type) if result_type else FloatType(32)
    sort = etype.to_z3_sort()

    if len(op.operands) == 1:
        arg = state.get_fp(op.operands[0]) if is_float_type(etype) else state.get(op.operands[0])
        fn = z3.Function(fn_name, sort, sort)
        state.set(res, fn(arg), result_type)
    elif len(op.operands) == 2:
        a = state.get(op.operands[0])
        b = state.get(op.operands[1])
        fn = z3.Function(fn_name, sort, sort, sort)
        state.set(res, fn(a, b), result_type)
    elif len(op.operands) >= 3:
        args = [state.get(o) for o in op.operands[:3]]
        fn = z3.Function(fn_name, sort, sort, sort, sort)
        state.set(res, fn(*args), result_type)
    else:
        state.set(res, make_z3_var(f"{fn_name}_{res}", etype), result_type)


# ---------------------------------------------------------------------------
# Dot product (matmul)
# ---------------------------------------------------------------------------

def _handle_dot(op: "Operation", state: "InterpreterState") -> None:
    """tt.dot %a, %b[, %c] – matrix multiply; result is a terminal."""
    if not op.results:
        return
    res = op.results[0]
    result_type = op.result_types[0] if op.result_types else FloatType(32)
    sym = make_z3_var(f"dot_{res}", result_type)
    state.set(res, sym, result_type)
    state.terminals.add(res)


# ---------------------------------------------------------------------------
# Atomic operations
# ---------------------------------------------------------------------------

def _handle_atomic(op: "Operation", state: "InterpreterState", kind: str) -> None:
    """Atomic ops – result is a fresh terminal (side-effect is opaque)."""
    if not op.results:
        return
    res = op.results[0]
    result_type = op.result_types[0] if op.result_types else FloatType(32)
    sym = make_z3_var(f"{kind}_{res}", result_type)
    state.set(res, sym, result_type)
    state.terminals.add(res)


# ---------------------------------------------------------------------------
# Triton-level casts
# ---------------------------------------------------------------------------

def _handle_tt_cast(op: "Operation", state: "InterpreterState") -> None:
    """tt.fp_to_fp / tt.int_to_fp / tt.fp_to_int."""
    if not op.operands or not op.results:
        return
    src = state.get(op.operands[0])
    result_type = op.result_types[0] if op.result_types else None

    if result_type:
        etype = element_type(result_type)
        target_sort = etype.to_z3_sort()
        if isinstance(target_sort, z3.FPSortRef) and z3.is_fp(src):
            state.set(op.results[0], z3.fpToFP(z3.RNE(), src, target_sort), result_type)
            return
        if isinstance(target_sort, z3.FPSortRef) and z3.is_bv(src):
            state.set(op.results[0], z3.fpSignedToFP(z3.RNE(), src, target_sort), result_type)
            return
        if not isinstance(target_sort, z3.FPSortRef) and z3.is_fp(src):
            w = target_sort.size() if hasattr(target_sort, "size") else 32
            state.set(op.results[0], z3.fpToSBV(z3.RTZ(), src, z3.BitVecSort(w)), result_type)
            return

    state.set(op.results[0], src, result_type)


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

def _handle_call(op: "Operation", state: "InterpreterState") -> None:
    """tt.call – treat results as fresh symbols."""
    for res in op.results:
        state.set_unknown(res, "tt.call_result")


# ---------------------------------------------------------------------------
# make_tensor_ptr
# ---------------------------------------------------------------------------

def _handle_make_tensor_ptr(op: "Operation", state: "InterpreterState") -> None:
    if not op.results:
        return
    res = op.results[0]
    if op.operands:
        state.set(op.results[0], state.get(op.operands[0]))
    else:
        state.set(res, z3.BitVec(f"tensor_ptr_{res}", 64))
