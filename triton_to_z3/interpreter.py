"""Core Z3 symbolic interpreter for TTIR.

Manages interpreter state and dispatches parsed operations to the
appropriate dialect handler.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

import z3

from .parser import Operation, Function, parse_ttir
from .types import (
    FloatType,
    IntegerType,
    MLIRType,
    element_type,
    is_float_type,
    is_int_type,
    make_z3_var,
)


# ---------------------------------------------------------------------------
# Interpreter state
# ---------------------------------------------------------------------------

@dataclass
class InterpreterState:
    """Holds all symbolic state during interpretation."""

    regs: dict[str, z3.ExprRef] = field(default_factory=dict)
    reg_types: dict[str, MLIRType] = field(default_factory=dict)
    terminals: set[str] = field(default_factory=set)
    stores: list[tuple[str, str]] = field(default_factory=list)
    _yield_stack: list[list[z3.ExprRef]] = field(default_factory=list)

    # -- Read / write helpers ---------------------------------------------

    def set(self, name: str, val: z3.ExprRef, mlir_type: MLIRType | None = None) -> None:
        self.regs[name] = val
        if mlir_type is not None:
            self.reg_types[name] = mlir_type

    def get(self, name: str) -> z3.ExprRef:
        if name in self.regs:
            return self.regs[name]
        # Auto-create as unknown
        sym = z3.FP(name, z3.Float32())
        self.regs[name] = sym
        return sym

    def has(self, name: str) -> bool:
        return name in self.regs

    def type_of(self, name: str) -> MLIRType | None:
        return self.reg_types.get(name)

    def set_unknown(self, name: str, context: str = "") -> None:
        """Create a fresh symbolic variable for an unknown / unsupported op."""
        sym = z3.FP(f"unknown_{context}_{name}", z3.Float32())
        self.regs[name] = sym

    # -- Typed getters (coerce if needed) ---------------------------------

    def get_fp(self, name: str) -> z3.ExprRef:
        """Get the value as a floating-point expression."""
        val = self.get(name)
        if z3.is_fp(val):
            return val
        # If it's a BV that should be float, create a symbolic FP alias
        return z3.FP(name, z3.Float32())

    def get_bv(self, name: str) -> z3.ExprRef:
        """Get the value as a bitvector expression."""
        val = self.get(name)
        if z3.is_bv(val):
            return val
        # If it's bool, convert to bv1
        if z3.is_bool(val):
            return z3.If(val, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))
        # Symbolic BV alias
        t = self.type_of(name)
        width = 32
        if t is not None:
            from .types import bv_width
            width = bv_width(t)
        return z3.BitVec(name, width)

    def get_bool(self, name: str) -> z3.ExprRef:
        """Get the value as a boolean expression."""
        val = self.get(name)
        if z3.is_bool(val):
            return val
        if z3.is_bv(val):
            return val == z3.BitVecVal(1, val.size())
        # Fallback
        return z3.Bool(name)

    # -- Yield management (for scf.yield / scf.for / scf.if) -------------

    def push_yield(self, vals: list[z3.ExprRef]) -> None:
        self._yield_stack.append(vals)

    def pop_yield(self) -> list[z3.ExprRef]:
        return self._yield_stack.pop() if self._yield_stack else []

    # -- Forking (for scf.if branch exploration) --------------------------

    def fork(self) -> InterpreterState:
        """Create a shallow copy for branch exploration."""
        return InterpreterState(
            regs=dict(self.regs),
            reg_types=dict(self.reg_types),
            terminals=set(self.terminals),
            stores=list(self.stores),
            _yield_stack=[],
        )


# ---------------------------------------------------------------------------
# Interpreter core
# ---------------------------------------------------------------------------

def interpret_ops(ops: list[Operation], state: InterpreterState) -> None:
    """Interpret a sequence of operations, mutating *state*."""
    from .dialects import dispatch

    for op in ops:
        dispatch(op, state)


def interpret_ttir(ir_text: str) -> tuple[dict[str, z3.ExprRef], set[str], InterpreterState]:
    """Parse and symbolically interpret a TTIR module.

    Returns ``(regs, terminals, state)`` where:
    - *regs* maps SSA names to Z3 expressions
    - *terminals* is the set of names that are "leaf" symbols (loads, args, etc.)
    - *state* is the full interpreter state (including stores)
    """
    func = parse_ttir(ir_text)
    state = InterpreterState()

    if func is None:
        return state.regs, state.terminals, state

    # Seed function arguments
    for arg_name, arg_type in func.args:
        sym = make_z3_var(arg_name, arg_type)
        state.set(arg_name, sym, arg_type)
        state.terminals.add(arg_name)

    # Execute body
    interpret_ops(func.body, state)

    return state.regs, state.terminals, state


# ---------------------------------------------------------------------------
# Expression chasing (deep substitution)
# ---------------------------------------------------------------------------

def deep_chase(
    expr: z3.ExprRef,
    regs: dict[str, z3.ExprRef],
    terminals: set[str],
    max_depth: int = 20,
) -> z3.ExprRef:
    """Repeatedly substitute non-terminal register definitions until only
    terminal symbols remain in *expr*.
    """
    current = expr
    for _ in range(max_depth):
        subs: list[tuple[z3.ExprRef, z3.ExprRef]] = []
        for name, val in regs.items():
            if name in terminals:
                continue
            # Build a symbol matching what the expression would reference
            if z3.is_fp(val):
                sym = z3.FP(name, val.sort())
            elif z3.is_bv(val):
                sym = z3.BitVec(name, val.size())
            elif z3.is_bool(val):
                sym = z3.Bool(name)
            else:
                continue
            subs.append((sym, val))

        if not subs:
            break

        try:
            new_expr = z3.substitute(current, *subs)
        except z3.Z3Exception:
            break

        if z3.eq(new_expr, current):
            break
        current = new_expr

    return current
