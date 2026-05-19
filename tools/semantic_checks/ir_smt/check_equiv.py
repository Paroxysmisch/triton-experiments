#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .kernel_ir import UnsupportedIR
from .recognizer import summarize_kernel
from .smt_emit import emit_equivalence_smt
from .ssa_parser import parse_module


def _summary(path: Path):
    return summarize_kernel(parse_module(path.read_text(encoding="utf-8")))


def _run_z3(smt_path: Path) -> dict[str, str | int]:
    z3_bin = shutil.which("z3")
    if z3_bin is None:
        return _run_python_z3(smt_path)
    result = subprocess.run(
        [z3_bin, "-smt2", str(smt_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "runner": z3_bin,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _run_python_z3(smt_path: Path) -> dict[str, str | int]:
    try:
        import z3
    except ImportError as exc:
        return {
            "runner": "python-z3",
            "returncode": 127,
            "stdout": "",
            "stderr": f"z3 executable not found and Python z3 import failed: {exc}",
        }

    solver = z3.Solver()
    smt = smt_path.read_text(encoding="utf-8")
    try:
        solver.from_string(smt)
        status = solver.check()
    except z3.Z3Exception as exc:
        return {
            "runner": "python-z3",
            "returncode": 1,
            "stdout": "",
            "stderr": str(exc),
        }

    stdout = str(status)
    if status == z3.sat:
        stdout += "\n" + str(solver.model())
    return {
        "runner": "python-z3",
        "returncode": 0,
        "stdout": stdout + "\n",
        "stderr": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check equivalence for tiny elementwise TTIR fragment")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--emit", type=Path)
    parser.add_argument("--run-z3", action="store_true")
    args = parser.parse_args()

    try:
        ref = _summary(args.reference)
        cand = _summary(args.candidate)
        smt = emit_equivalence_smt(ref, cand)
    except UnsupportedIR as exc:
        print(json.dumps({"status": "unsupported", "reason": str(exc)}, indent=2))
        return 2

    if args.emit:
        args.emit.write_text(smt, encoding="utf-8")

    report = {
        "status": "emitted",
        "fragment": ref.fragment,
        "reference_output": ref.output,
        "candidate_output": cand.output,
        "smt_path": str(args.emit) if args.emit else None,
    }
    if args.run_z3:
        if not args.emit:
            print(json.dumps({"status": "error", "reason": "--run-z3 requires --emit"}, indent=2))
            return 2
        report["z3"] = _run_z3(args.emit)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
