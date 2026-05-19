from __future__ import annotations

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
)


def _idx(expr: IndexExpr) -> str:
    if isinstance(expr, IndexConst):
        return str(expr.value)
    if isinstance(expr, IndexSymbol):
        return expr.name
    if isinstance(expr, IndexAdd):
        return f"(+ {_idx(expr.left)} {_idx(expr.right)})"
    if isinstance(expr, IndexMul):
        return f"(* {_idx(expr.left)} {_idx(expr.right)})"
    raise UnsupportedIR(f"cannot emit index expression: {expr!r}")


def _bool(expr: BoolExpr) -> str:
    if isinstance(expr, BoolLt):
        return f"(< {_idx(expr.left)} {_idx(expr.right)})"
    if isinstance(expr, BoolLe):
        return f"(<= {_idx(expr.left)} {_idx(expr.right)})"
    raise UnsupportedIR(f"cannot emit bool expression: {expr!r}")


def _val(expr: ValueExpr, index_name: str = "i") -> str:
    if isinstance(expr, ValueConst):
        return expr.value
    if isinstance(expr, ValueScalar):
        return expr.name
    if isinstance(expr, ValueLoad):
        # In elementwise_1d_v0, all accepted loads are at the canonical output
        # index. Emit in terms of the logical output index for equivalence.
        return f"(select {expr.tensor} {index_name})"
    if isinstance(expr, ValueBinary):
        return f"({expr.op} {_val(expr.left, index_name)} {_val(expr.right, index_name)})"
    if isinstance(expr, ValueUnary):
        inner = _val(expr.arg, index_name)
        if expr.op == "sqrt":
            # Non-linear sqrt is not in AUFLIRA. Keep a symbolic uninterpreted
            # function for structural equivalence checks.
            return f"(sqrtR {inner})"
        if expr.op == "tanh":
            return f"(tanhR {inner})"
    raise UnsupportedIR(f"cannot emit value expression: {expr!r}")


def _collect_arrays(expr: ValueExpr) -> set[str]:
    if isinstance(expr, ValueLoad):
        return {expr.tensor}
    if isinstance(expr, ValueBinary):
        return _collect_arrays(expr.left) | _collect_arrays(expr.right)
    if isinstance(expr, ValueUnary):
        return _collect_arrays(expr.arg)
    return set()


def _collect_scalars(expr: ValueExpr) -> set[str]:
    if isinstance(expr, ValueScalar):
        return {expr.name}
    if isinstance(expr, ValueBinary):
        return _collect_scalars(expr.left) | _collect_scalars(expr.right)
    if isinstance(expr, ValueUnary):
        return _collect_scalars(expr.arg)
    return set()


def emit_equivalence_smt(reference: KernelSummary, candidate: KernelSummary) -> str:
    arrays = sorted(_collect_arrays(reference.value) | _collect_arrays(candidate.value))
    scalars = sorted(_collect_scalars(reference.value) | _collect_scalars(candidate.value))
    lines: list[str] = [
        "(set-logic AUFLIRA)",
        "(declare-const N Int)",
        "(declare-const BLOCK Int)",
        "(assert (> N 0))",
        "(assert (> BLOCK 0))",
    ]
    for name in arrays:
        lines.append(f"(declare-const {name} (Array Int Real))")
    for name in scalars:
        # SSA leftovers are not meaningful scalars in the trusted fragment. They
        # can occur only for explicit scalar operands in hand-written summaries.
        if name not in {"pid0", "lane", "N", "BLOCK"}:
            lines.append(f"(declare-const {name} Real)")
    if "sqrt" in repr(reference.value) or "sqrt" in repr(candidate.value):
        lines.append("(declare-fun sqrtR (Real) Real)")
    if "tanh" in repr(reference.value) or "tanh" in repr(candidate.value):
        lines.append("(declare-fun tanhR (Real) Real)")
    lines.extend(
        [
            "",
            f"(define-fun ref_value ((i Int)) Real {_val(reference.value)})",
            f"(define-fun cand_value ((i Int)) Real {_val(candidate.value)})",
            "",
            "(assert",
            "  (exists ((i Int))",
            "    (and (<= 0 i)",
            "         (< i N)",
            "         (not (= (ref_value i) (cand_value i))))))",
            "",
            "(check-sat)",
            "(get-model)",
        ]
    )
    return "\n".join(lines) + "\n"


def emit_memory_safety_smt(summary: KernelSummary) -> str:
    index = _idx(summary.index)
    active = _bool(summary.mask)
    return "\n".join(
        [
            "(set-logic LIA)",
            "(declare-const N Int)",
            "(declare-const BLOCK Int)",
            "(assert (> N 0))",
            "(assert (> BLOCK 0))",
            "(assert",
            "  (not",
            "    (forall ((pid0 Int) (lane Int))",
            "      (=> (and (<= 0 pid0) (<= 0 lane) (< lane BLOCK)",
            f"               {active})",
            f"          (and (<= 0 {index}) (< {index} N))))))",
            "(check-sat)",
        ]
    ) + "\n"

