"""Triton GPU (``ttg.*``) dialect handler.

These ops are GPU-specific: async copies, shared-memory management, layout
conversions, warp specialisation, and pipelining primitives.  For symbolic
execution the strategy is:

**Value-preserving (aliases)**
    ``convert_layout``, ``memdesc_reinterpret``, ``memdesc_reshape``,
    ``memdesc_subslice``, ``memdesc_trans``, ``memdesc_index``
    – the symbolic value is unchanged; these only affect physical layout.

**Memory reads → fresh terminal symbols**
    ``local_load``, ``local_gather``, ``async_copy_global_to_local``
    – the loaded value is opaque, modelled as a fresh Z3 variable.

**Memory writes → recorded side-effects**
    ``local_store``, ``local_scatter``, ``local_atomic_scatter_rmw``
    – stored values are recorded so ``deep_chase`` can reach them.

**Pure side-effects / synchronisation → no-ops**
    ``barrier``, ``async_wait``, ``async_commit_group``,
    ``local_dealloc``, ``warp_return``, ``mask.return``

**Hardware queries → terminal symbols**
    ``warp_id``, ``predicate_stage``

**Allocation → fresh pointer symbol**
    ``global_scratch_alloc``, ``local_alloc``

**Type conversion**
    ``fp4_to_fp`` – modelled as an uninterpreted cast.

**Warp / region ops**
    ``warp_specialize`` – default region is executed symbolically.
    ``mask`` – region is executed (condition is symbolic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import z3

from ..types import (
    FloatType,
    IntegerType,
    PointerType,
    element_type as get_element_type,
    is_float_type,
    make_z3_var,
)

if TYPE_CHECKING:
    from ..interpreter import InterpreterState
    from ..parser import Operation


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def handle(op: "Operation", state: "InterpreterState") -> None:
    name = op.op.removeprefix("ttg.")
    match name:
        # --- Layout / descriptor transforms (value aliases) ---------------
        case "convert_layout":
            _handle_alias(op, state)
        case "memdesc_reinterpret":
            _handle_alias(op, state)
        case "memdesc_reshape":
            _handle_alias(op, state)
        case "memdesc_subslice":
            _handle_alias(op, state)
        case "memdesc_trans":
            _handle_alias(op, state)
        case "memdesc_index":
            _handle_memdesc_index(op, state)

        # --- Local memory reads (fresh terminals) -------------------------
        case "local_load":
            _handle_local_load(op, state)
        case "local_gather":
            _handle_local_gather(op, state)
        case "async_copy_global_to_local":
            _handle_async_copy(op, state)

        # --- Local memory writes (side-effects) ---------------------------
        case "local_store":
            _handle_local_store(op, state)
        case "local_scatter":
            _handle_local_scatter(op, state)
        case "local_atomic_scatter_rmw":
            _handle_local_atomic(op, state)

        # --- Synchronisation / side-effects (no-ops) ----------------------
        case "barrier":
            pass
        case "async_wait":
            _handle_token_passthrough(op, state)
        case "async_commit_group":
            _handle_token_passthrough(op, state)
        case "local_dealloc":
            pass

        # --- Allocation ---------------------------------------------------
        case "global_scratch_alloc":
            _handle_alloc(op, state, "global_scratch")
        case "local_alloc":
            _handle_local_alloc(op, state)

        # --- Hardware queries (terminals) ---------------------------------
        case "warp_id":
            _handle_warp_id(op, state)
        case "predicate_stage":
            _handle_predicate_stage(op, state)

        # --- Type conversion ----------------------------------------------
        case "fp4_to_fp":
            _handle_fp4_to_fp(op, state)

        # --- Warp specialisation / mask regions ---------------------------
        case "warp_specialize":
            _handle_warp_specialize(op, state)
        case "warp_specialize.partitions":
            pass  # Container; partitions handled by warp_specialize
        case "warp_return":
            pass  # Implicit terminator
        case "warp_yield":
            _handle_warp_yield(op, state)
        case "mask":
            _handle_mask(op, state)
        case "mask.return":
            _handle_mask_return(op, state)

        # --- Fallback -----------------------------------------------------
        case _:
            for r in op.results:
                state.set_unknown(r, op.op)


# ---------------------------------------------------------------------------
# Value aliases (layout / descriptor transforms)
# ---------------------------------------------------------------------------


def _handle_alias(op: "Operation", state: "InterpreterState") -> None:
    """Layout conversions and descriptor reshapes don't change the symbolic value."""
    if not op.results:
        return
    if op.operands:
        src = state.get(op.operands[0])
        result_type = op.result_types[0] if op.result_types else None
        state.set(op.results[0], src, result_type)
    else:
        state.set_unknown(op.results[0], op.op)


def _handle_memdesc_index(op: "Operation", state: "InterpreterState") -> None:
    """Subview of a memory descriptor — symbolically the same value."""
    if not op.results:
        return
    if op.operands:
        state.set(op.results[0], state.get(op.operands[0]))
    else:
        state.set_unknown(op.results[0], "memdesc_index")


# ---------------------------------------------------------------------------
# Local memory reads → fresh terminal symbols
# ---------------------------------------------------------------------------


def _handle_local_load(op: "Operation", state: "InterpreterState") -> None:
    """Load from local (shared) memory — produces a fresh terminal."""
    if not op.results:
        return
    res = op.results[0]
    result_type = op.result_types[0] if op.result_types else None
    etype = get_element_type(result_type) if result_type else FloatType(32)
    sym = make_z3_var(f"local_load_{res}", etype)
    state.set(res, sym, result_type)
    state.terminals.add(res)


def _handle_local_gather(op: "Operation", state: "InterpreterState") -> None:
    """Gather from shared memory at indices — produces a fresh terminal."""
    if not op.results:
        return
    res = op.results[0]
    result_type = op.result_types[0] if op.result_types else None
    etype = get_element_type(result_type) if result_type else FloatType(32)
    sym = make_z3_var(f"local_gather_{res}", etype)
    state.set(res, sym, result_type)
    state.terminals.add(res)


def _handle_async_copy(op: "Operation", state: "InterpreterState") -> None:
    """Async copy global→local.  The token result is a placeholder; the
    actual data is only available after ``async_wait``."""
    # Token result
    for res in op.results:
        sym = z3.BitVec(f"async_token_{res}", 32)
        state.set(res, sym)


# ---------------------------------------------------------------------------
# Local memory writes → recorded side-effects
# ---------------------------------------------------------------------------


def _handle_local_store(op: "Operation", state: "InterpreterState") -> None:
    """Store a tensor into shared memory."""
    # local_store %src, %dst
    if len(op.operands) >= 2:
        state.stores.append((op.operands[1], op.operands[0]))


def _handle_local_scatter(op: "Operation", state: "InterpreterState") -> None:
    """Scatter elements into shared memory at indices."""
    # local_scatter %dst, %values, %indices
    if len(op.operands) >= 2:
        state.stores.append((op.operands[0], op.operands[1]))


def _handle_local_atomic(op: "Operation", state: "InterpreterState") -> None:
    """Atomic scatter RMW into shared memory — returns the previous values."""
    # Record the write side-effect
    if len(op.operands) >= 2:
        state.stores.append((op.operands[0], op.operands[1]))
    # The result (previous values) is a fresh terminal
    if op.results:
        res = op.results[0]
        result_type = op.result_types[0] if op.result_types else None
        etype = get_element_type(result_type) if result_type else FloatType(32)
        sym = make_z3_var(f"atomic_prev_{res}", etype)
        state.set(res, sym, result_type)
        state.terminals.add(res)


# ---------------------------------------------------------------------------
# Tokens (async_wait / async_commit_group)
# ---------------------------------------------------------------------------


def _handle_token_passthrough(op: "Operation", state: "InterpreterState") -> None:
    """Async tokens are opaque — create a placeholder for each result."""
    for res in op.results:
        sym = z3.BitVec(f"async_token_{res}", 32)
        state.set(res, sym)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def _handle_alloc(op: "Operation", state: "InterpreterState", kind: str) -> None:
    """Allocate a buffer — returns a fresh pointer symbol."""
    if not op.results:
        return
    res = op.results[0]
    sym = z3.BitVec(f"{kind}_ptr_{res}", 64)
    state.set(res, sym, PointerType(FloatType(32)))
    state.terminals.add(res)


def _handle_local_alloc(op: "Operation", state: "InterpreterState") -> None:
    """Allocate shared memory.  If an initialiser tensor is provided,
    the descriptor is an alias of that value; otherwise a fresh symbol."""
    if not op.results:
        return
    res = op.results[0]
    if op.operands:
        # Initialised from a source tensor
        state.set(op.results[0], state.get(op.operands[0]))
    else:
        result_type = op.result_types[0] if op.result_types else None
        etype = get_element_type(result_type) if result_type else FloatType(32)
        sym = make_z3_var(f"local_alloc_{res}", etype)
        state.set(res, sym, result_type)
        state.terminals.add(res)


# ---------------------------------------------------------------------------
# Hardware queries
# ---------------------------------------------------------------------------


def _handle_warp_id(op: "Operation", state: "InterpreterState") -> None:
    if not op.results:
        return
    res = op.results[0]
    sym = z3.BitVec("warp_id", 32)
    state.set(res, sym, IntegerType(32))
    state.terminals.add(res)


def _handle_predicate_stage(op: "Operation", state: "InterpreterState") -> None:
    """Pipeline stage predicate — a symbolic boolean."""
    if not op.results:
        return
    res = op.results[0]
    stage = op.attributes.get("stage", "?")
    sym = z3.Bool(f"predicate_stage_{stage}")
    state.set(res, sym, IntegerType(1))
    state.terminals.add(res)


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


def _handle_fp4_to_fp(op: "Operation", state: "InterpreterState") -> None:
    """Upcast fp4 (e2m1) packed in i8 to floating-point.

    Modelled as an uninterpreted function since Z3 has no native fp4.
    """
    if not op.operands or not op.results:
        return
    res = op.results[0]
    src = state.get(op.operands[0])
    result_type = op.result_types[0] if op.result_types else FloatType(32)
    etype = get_element_type(result_type)
    sort = etype.to_z3_sort()
    fn = z3.Function("fp4_to_fp", src.sort(), sort)
    state.set(res, fn(src), result_type)


# ---------------------------------------------------------------------------
# Warp specialisation / mask regions
# ---------------------------------------------------------------------------


def _handle_warp_specialize(op: "Operation", state: "InterpreterState") -> None:
    """Execute the *default* warp-group region symbolically.

    Partition regions are architecture-specific and run in parallel;
    for symbolic analysis we only trace the default (first) region.
    """
    from ..interpreter import interpret_ops

    if op.regions:
        interpret_ops(op.regions[0].body, state)

    # Collect yielded values for the default passthrough results
    yield_vals = state.pop_yield()
    for i, res in enumerate(op.results):
        if i < len(yield_vals):
            state.set(res, yield_vals[i])
        else:
            state.set_unknown(res, "warp_specialize_result")


def _handle_warp_yield(op: "Operation", state: "InterpreterState") -> None:
    """Yield from the default region of ``warp_specialize``."""
    vals = [state.get(name) for name in op.operands]
    state.push_yield(vals)


def _handle_mask(op: "Operation", state: "InterpreterState") -> None:
    """Conditional execution for pipelining — execute the region
    under a symbolic predicate."""
    from ..interpreter import interpret_ops

    if op.regions:
        interpret_ops(op.regions[0].body, state)

    # Collect results from mask.return
    yield_vals = state.pop_yield()
    for i, res in enumerate(op.results):
        if i < len(yield_vals):
            state.set(res, yield_vals[i])
        else:
            state.set_unknown(res, "mask_result")


def _handle_mask_return(op: "Operation", state: "InterpreterState") -> None:
    """Terminator for ``ttg.mask`` — push values for the parent to collect."""
    vals = [state.get(name) for name in op.operands]
    state.push_yield(vals)
