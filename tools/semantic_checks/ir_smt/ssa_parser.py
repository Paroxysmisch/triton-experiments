from __future__ import annotations

import re
from dataclasses import dataclass, field

from .kernel_ir import UnsupportedIR


SSA_RE = re.compile(r"%[A-Za-z_.$-][A-Za-z0-9_.$-]*|%[0-9]+")
OP_RE = re.compile(r'"?([A-Za-z_][A-Za-z0-9_.-]*)"?')


@dataclass(frozen=True)
class Operation:
    result_ids: tuple[str, ...]
    op_name: str
    operands: tuple[str, ...]
    attrs: dict[str, str]
    raw: str


@dataclass
class Module:
    ops: list[Operation] = field(default_factory=list)
    defs: dict[str, Operation] = field(default_factory=dict)
    uses: dict[str, list[Operation]] = field(default_factory=dict)

    def add_op(self, op: Operation) -> None:
        self.ops.append(op)
        for result in op.result_ids:
            if result in self.defs:
                raise UnsupportedIR(f"SSA value defined twice: {result}")
            self.defs[result] = op
        for operand in op.operands:
            self.uses.setdefault(operand, []).append(op)


def _strip_comments(line: str) -> str:
    # MLIR comments start with //; keep this intentionally simple.
    return line.split("//", 1)[0].strip()


def _split_results(line: str) -> tuple[tuple[str, ...], str]:
    if "=" not in line:
        return (), line.strip()
    lhs, rhs = line.split("=", 1)
    result_ids = tuple(SSA_RE.findall(lhs))
    return result_ids, rhs.strip()


def _parse_attrs(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,\]}):]+)", text):
        attrs[key] = value.strip().strip('"')
    return attrs


def parse_module(text: str) -> Module:
    module = Module()
    for raw_line in text.splitlines():
        line = _strip_comments(raw_line)
        if not line or line in {"{", "}"}:
            continue
        if line.startswith(("module", "func.func", "tt.func", "return")):
            continue
        result_ids, rhs = _split_results(line)
        match = OP_RE.match(rhs)
        if not match:
            continue
        op_name = match.group(1)
        operands = tuple(v for v in SSA_RE.findall(rhs) if v not in result_ids)
        attrs = _parse_attrs(rhs)
        module.add_op(Operation(result_ids, op_name, operands, attrs, raw_line.strip()))
    return module

