"""triton_to_z3 – Symbolic Z3 interpreter for Triton IR (TTIR).

Usage::

    from triton_to_z3 import interpret_ttir, deep_chase

    regs, terminals, state = interpret_ttir(ir_text)
    formula = deep_chase(regs[target], regs, terminals)
"""

from .interpreter import deep_chase, interpret_ttir, InterpreterState

__all__ = ["deep_chase", "interpret_ttir", "InterpreterState"]
