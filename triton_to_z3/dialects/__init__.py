"""Dialect handler registry.

Each dialect module exposes a ``handle(op, state)`` function that interprets
one MLIR operation and mutates *state* in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import arith, math_ops, scf, triton

if TYPE_CHECKING:
    from ..interpreter import InterpreterState
    from ..parser import Operation

DIALECT_HANDLERS: dict[str, "HandleFn"] = {
    "arith": arith.handle,
    "math": math_ops.handle,
    "scf": scf.handle,
    "tt": triton.handle,
}

type HandleFn = type(arith.handle)


def dispatch(op: "Operation", state: "InterpreterState") -> None:
    """Route *op* to its dialect handler."""
    dialect = op.op.split(".")[0]
    handler = DIALECT_HANDLERS.get(dialect)
    if handler is not None:
        handler(op, state)
    else:
        # Unknown dialect – record a placeholder for each result
        for r in op.results:
            state.set_unknown(r, op.op)
