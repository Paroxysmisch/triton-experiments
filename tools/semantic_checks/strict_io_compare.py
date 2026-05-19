#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


RESULT_GLOBALS = ("test_results", "result_gold", "results")


def _run_stdout(path: Path, timeout: float) -> dict[str, Any]:
    # Run from an isolated temp directory so sibling generated files such as
    # torch.py do not shadow real installed packages.
    with tempfile.TemporaryDirectory(prefix="strict_io_stdout_") as tmp:
        staged = Path(tmp) / "candidate_under_test.py"
        shutil.copy2(path, staged)
        result = subprocess.run(
            [sys.executable, str(staged)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _seed_everything(seed: int) -> None:
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _import_module(path: Path, timeout: float, seed: int) -> dict[str, Any]:
    # Import from an isolated temp directory so sibling generated files such as
    # torch.py do not shadow real installed packages.
    del timeout
    tmp_ctx = tempfile.TemporaryDirectory(prefix="strict_io_import_")
    tmp = Path(tmp_ctx.name)
    staged = tmp / "candidate_under_test.py"
    shutil.copy2(path, staged)
    module_name = f"_strict_io_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, staged)
    if spec is None or spec.loader is None:
        tmp_ctx.cleanup()
        return {"ok": False, "error": f"cannot import {path}"}
    module = importlib.util.module_from_spec(spec)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            _seed_everything(seed)
            spec.loader.exec_module(module)
    except BaseException:
        tmp_ctx.cleanup()
        return {
            "ok": False,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "error": traceback.format_exc(),
        }
    for name in RESULT_GLOBALS:
        if hasattr(module, name):
            result = {
                "ok": True,
                "result_name": name,
                "result": getattr(module, name),
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
            }
            tmp_ctx.cleanup()
            return result
    result = {
        "ok": True,
        "result_name": None,
        "result": None,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
    }
    tmp_ctx.cleanup()
    return result


def _is_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"


def _normalize(value: Any) -> Any:
    if _is_tensor(value):
        tensor = value.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return {
            "__tensor__": True,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "values": tensor.tolist(),
        }
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    return repr(value)


def _compare_normalized(left: Any, right: Any, path: str = "$") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict) and left.get("__tensor__") and right.get("__tensor__"):
        mismatches: list[str] = []
        if left["shape"] != right["shape"]:
            mismatches.append(f"{path}: tensor shape differs: {left['shape']} != {right['shape']}")
        if left["dtype"] != right["dtype"]:
            mismatches.append(f"{path}: tensor dtype differs: {left['dtype']} != {right['dtype']}")
        if left["values"] != right["values"]:
            mismatches.append(f"{path}: tensor values differ")
        return mismatches
    if type(left) is not type(right):
        return [f"{path}: type differs: {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, dict):
        mismatches = []
        if set(left) != set(right):
            mismatches.append(f"{path}: keys differ: {sorted(left)} != {sorted(right)}")
        for key in sorted(set(left) & set(right)):
            mismatches.extend(_compare_normalized(left[key], right[key], f"{path}.{key}"))
        return mismatches
    if isinstance(left, list):
        mismatches = []
        if len(left) != len(right):
            mismatches.append(f"{path}: length differs: {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            mismatches.extend(_compare_normalized(left_item, right_item, f"{path}[{index}]"))
        return mismatches
    return [] if left == right else [f"{path}: value differs: {left!r} != {right!r}"]


def compare(candidate: Path, reference: Path, timeout: float, seed: int) -> dict[str, Any]:
    cand_stdout = _run_stdout(candidate, timeout)
    ref_stdout = _run_stdout(reference, timeout)
    stdout_equal = cand_stdout["stdout"] == ref_stdout["stdout"]

    cand_import = _import_module(candidate, timeout, seed)
    ref_import = _import_module(reference, timeout, seed)
    report: dict[str, Any] = {
        "candidate": str(candidate),
        "reference": str(reference),
        "seed": seed,
        "repo_stdout_equal": stdout_equal,
        "candidate_stdout": cand_stdout,
        "reference_stdout": ref_stdout,
        "candidate_import_ok": cand_import["ok"],
        "reference_import_ok": ref_import["ok"],
    }

    if not cand_import["ok"] or not ref_import["ok"]:
        report["strict_status"] = "import_error"
        report["candidate_import_error"] = cand_import.get("error")
        report["reference_import_error"] = ref_import.get("error")
        return report

    cand_norm = _normalize(cand_import["result"])
    ref_norm = _normalize(ref_import["result"])
    mismatches = _compare_normalized(cand_norm, ref_norm)
    report.update(
        {
            "candidate_result_name": cand_import["result_name"],
            "reference_result_name": ref_import["result_name"],
            "strict_status": "pass" if not mismatches else "fail",
            "strict_mismatches": mismatches,
            "stdout_false_positive": stdout_equal and bool(mismatches),
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare TritonBench candidate/reference outputs beyond stdout.")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    report = compare(args.candidate, args.reference, args.timeout, args.seed)
    print(json.dumps(report, indent=2))
    if report.get("stdout_false_positive"):
        return 1
    if report.get("strict_status") not in {"pass"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
