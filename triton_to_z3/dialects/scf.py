"""SCF (Structured Control Flow) dialect handler.

Symbolic-execution strategy:
- **scf.for**  – execute the loop body *once* symbolically; loop-carried
  values become fresh Z3 symbols before the iteration.
- **scf.if**   – model with Z3 ``If`` over both branches.
- **scf.while** – execute the *before* region once; the result is a fresh
  symbol (full loop semantics are undecidable in general).
- **scf.yield / scf.condition** – propagated by parent-op handlers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import z3

from ..types import make_z3_var

if TYPE_CHECKING:
    from ..interpreter import InterpreterState
    from ..parser import Operation


def handle(op: "Operation", state: "InterpreterState") -> None:
    name = op.op.removeprefix("scf.")
    match name:
        case "for":
            _handle_for(op, state)
        case "if":
            _handle_if(op, state)
        case "while":
            _handle_while(op, state)
        case "yield":
            _handle_yield(op, state)
        case "condition":
            pass  # Handled implicitly by while
        case "index_switch":
            _handle_index_switch(op, state)
        case "execute_region":
            _handle_execute_region(op, state)
        case "forall":
            _handle_forall(op, state)
        case "parallel":
            _handle_parallel(op, state)
        case "reduce":
            pass  # Body handled by scf.parallel
        case "reduce.return":
            pass
        case "forall.in_parallel":
            pass
        case _:
            for r in op.results:
                state.set_unknown(r, op.op)


# ---------------------------------------------------------------------------
# scf.for
# ---------------------------------------------------------------------------


def _handle_for(op: "Operation", state: "InterpreterState") -> None:
    """Execute the loop body once symbolically.

    The induction variable and any iter_args are modelled as fresh symbols.
    After executing the body, the yielded values (if any) are propagated to
    the for-op's results.
    """
    from ..interpreter import interpret_ops

    # Induction variable — use the for-loop's type or fall back to i32
    iv_name = op.attributes.get("iv")
    if iv_name:
        from ..types import IntegerType

        iv_type = op.result_types[0] if op.result_types else IntegerType(32)
        state.set(iv_name, make_z3_var(iv_name, iv_type), iv_type)

    # iter_args: in real MLIR these are extra operands mapped to region block
    # args.  For our symbolic model, we set them as fresh symbols.
    if op.regions:
        region = op.regions[0]
        for arg_name, arg_type in region.args:
            if not state.has(arg_name):
                state.set(arg_name, make_z3_var(arg_name, arg_type), arg_type)

        # Execute body
        interpret_ops(region.body, state)

    # After body execution, yielded values (stored by scf.yield handler)
    # should map to the for-op result names.
    yield_vals = state.pop_yield()
    for i, res in enumerate(op.results):
        if i < len(yield_vals):
            state.set(res, yield_vals[i])
        else:
            state.set_unknown(res, "scf.for_result")


# ---------------------------------------------------------------------------
# scf.if
# ---------------------------------------------------------------------------


def _handle_if(op: "Operation", state: "InterpreterState") -> None:
    """Model both branches and combine with ``z3.If``."""
    from ..interpreter import interpret_ops

    if not op.operands:
        return

    cond = state.get_bool(op.operands[0])

    then_vals: list[z3.ExprRef] = []
    else_vals: list[z3.ExprRef] = []

    # Then branch
    if len(op.regions) >= 1:
        then_state = state.fork()
        interpret_ops(op.regions[0].body, then_state)
        then_vals = then_state.pop_yield()

    # Else branch
    if len(op.regions) >= 2:
        else_state = state.fork()
        interpret_ops(op.regions[1].body, else_state)
        else_vals = else_state.pop_yield()

    # Merge results
    for i, res in enumerate(op.results):
        t_val = then_vals[i] if i < len(then_vals) else None
        e_val = else_vals[i] if i < len(else_vals) else None
        if t_val is not None and e_val is not None:
            state.set(res, z3.If(cond, t_val, e_val))
        elif t_val is not None:
            state.set(res, t_val)
        elif e_val is not None:
            state.set(res, e_val)
        else:
            state.set_unknown(res, "scf.if_result")


# ---------------------------------------------------------------------------
# scf.while
# ---------------------------------------------------------------------------


def _handle_while(op: "Operation", state: "InterpreterState") -> None:
    """Execute *before* region once; result is a fresh symbol."""
    from ..interpreter import interpret_ops

    # Set region block args from operands
    if op.regions:
        before = op.regions[0]
        for idx, (arg_name, arg_type) in enumerate(before.args):
            if idx < len(op.operands):
                state.set(arg_name, state.get(op.operands[idx]), arg_type)
            else:
                state.set(arg_name, make_z3_var(arg_name, arg_type), arg_type)

        interpret_ops(before.body, state)

    # After region (if present)
    if len(op.regions) >= 2:
        after = op.regions[1]
        for arg_name, arg_type in after.args:
            if not state.has(arg_name):
                state.set(arg_name, make_z3_var(arg_name, arg_type), arg_type)
        interpret_ops(after.body, state)

    # Results are fresh symbols (loop semantics are opaque)
    for res in op.results:
        state.set_unknown(res, "scf.while_result")


# ---------------------------------------------------------------------------
# scf.yield
# ---------------------------------------------------------------------------


def _handle_yield(op: "Operation", state: "InterpreterState") -> None:
    """Store yielded values for the parent op to pick up."""
    vals = [state.get(name) for name in op.operands]
    state.push_yield(vals)


# ---------------------------------------------------------------------------
# Less common SCF ops
# ---------------------------------------------------------------------------


def _handle_index_switch(op: "Operation", state: "InterpreterState") -> None:
    from ..interpreter import interpret_ops

    # Execute default region; results are fresh symbols
    if op.regions:
        interpret_ops(op.regions[0].body, state)
    for res in op.results:
        if not state.has(res):
            state.set_unknown(res, "scf.index_switch_result")


def _handle_execute_region(op: "Operation", state: "InterpreterState") -> None:
    from ..interpreter import interpret_ops

    if op.regions:
        interpret_ops(op.regions[0].body, state)
        yield_vals = state.pop_yield()
        for i, res in enumerate(op.results):
            if i < len(yield_vals):
                state.set(res, yield_vals[i])
            else:
                state.set_unknown(res, "scf.execute_region_result")


def _handle_forall(op: "Operation", state: "InterpreterState") -> None:
    """Treat like scf.for – single symbolic iteration."""
    from ..interpreter import interpret_ops

    if op.regions:
        for arg_name, arg_type in op.regions[0].args:
            if not state.has(arg_name):
                state.set(arg_name, make_z3_var(arg_name, arg_type), arg_type)
        interpret_ops(op.regions[0].body, state)

    for res in op.results:
        if not state.has(res):
            state.set_unknown(res, "scf.forall_result")


def _handle_parallel(op: "Operation", state: "InterpreterState") -> None:
    """Treat like scf.for – single symbolic iteration."""
    from ..interpreter import interpret_ops

    if op.regions:
        for arg_name, arg_type in op.regions[0].args:
            if not state.has(arg_name):
                state.set(arg_name, make_z3_var(arg_name, arg_type), arg_type)
        interpret_ops(op.regions[0].body, state)

    for res in op.results:
        if not state.has(res):
            state.set_unknown(res, "scf.parallel_result")
