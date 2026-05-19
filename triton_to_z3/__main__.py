"""CLI entry point: ``python -m triton_to_z3 [file]``

Reads TTIR from *file* (or stdin) and prints the Z3 symbolic formula for
every ``tt.store`` target found in the IR.
"""

from __future__ import annotations

import re
import sys

from .interpreter import deep_chase, interpret_ttir


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        with open(sys.argv[1]) as f:
            ir_text = f.read()
    else:
        ir_text = sys.stdin.read()

    regs, terminals, state = interpret_ttir(ir_text)

    if not regs:
        print(";; No operations interpreted.", file=sys.stderr)
        return

    # Print formulas for every stored value
    if state.stores:
        for ptr_name, val_name in state.stores:
            if val_name in regs:
                formula = deep_chase(regs[val_name], regs, terminals)
                print(f";; Formula for tt.store → %{val_name}:")
                print(formula)
                print()
    else:
        # Fallback: scan raw IR for tt.store patterns (backwards compat)
        store_re = re.compile(r"tt\.store\s+\S+,\s+%(?P<val>\w+)")
        for m in store_re.finditer(ir_text):
            val_name = m.group("val")
            if val_name in regs:
                formula = deep_chase(regs[val_name], regs, terminals)
                print(f";; Formula for tt.store → %{val_name}:")
                print(formula)
                print()

    # Also print a summary of terminals
    print(";; Terminal symbols:")
    for t in sorted(terminals):
        if t in regs:
            print(f";;   %{t} = {regs[t]}")


if __name__ == "__main__":
    main()
