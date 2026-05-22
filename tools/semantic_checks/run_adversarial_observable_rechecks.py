#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable


MARKER = "#" * 146


def _implementation_prefix(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if MARKER in text:
        return text.split(MARKER, 1)[0].strip() + "\n"
    return text


def _load_module_from_source(source: str, name: str, tmp_dir: Path) -> Any:
    path = tmp_dir / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path = [item for item in sys.path if item != str(tmp_dir)]
        spec.loader.exec_module(module)
    finally:
        sys.path = old_path
    return module


def _candidate_source(candidate: Path) -> str:
    return _implementation_prefix(candidate)


def _reference_source(reference: Path) -> str:
    return _implementation_prefix(reference)


def _seed(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_close(left: Any, right: Any, *, atol: float = 1e-4, rtol: float = 1e-4) -> None:
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol, check_dtype=False)
        return
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        assert len(left) == len(right), f"length differs: {len(left)} != {len(right)}"
        for l_item, r_item in zip(left, right):
            _assert_close(l_item, r_item, atol=atol, rtol=rtol)
        return
    assert left == right, f"value differs: {left!r} != {right!r}"


def _case_relu_batch_norm_conv2d(candidate: Any, reference: Any, seed: int) -> list[str]:
    import torch

    _seed(seed)
    checks: list[str] = []
    cases = [
        {"shape": (2, 3, 9, 11), "out_channels": 4, "kernel": 3, "stride": 1, "padding": 1, "bias": True, "training": False},
        {"shape": (1, 2, 7, 5), "out_channels": 2, "kernel": 1, "stride": 1, "padding": 0, "bias": False, "training": False},
        {"shape": (3, 4, 12, 10), "out_channels": 8, "kernel": 3, "stride": 2, "padding": 1, "bias": True, "training": True},
    ]
    for index, case in enumerate(cases, 1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(case["shape"], device=device)
        weight = torch.randn(case["out_channels"], case["shape"][1], case["kernel"], case["kernel"], device=device)
        bias = torch.randn(case["out_channels"], device=device) if case["bias"] else None
        running_mean_c = torch.zeros(case["out_channels"], device=device)
        running_var_c = torch.ones(case["out_channels"], device=device)
        running_mean_r = running_mean_c.clone()
        running_var_r = running_var_c.clone()
        kwargs = dict(
            input=x,
            weight=weight,
            bias=bias,
            stride=case["stride"],
            padding=case["padding"],
            running_mean=running_mean_c,
            running_var=running_var_c,
            bn_weight=torch.randn(case["out_channels"], device=device),
            bn_bias=torch.randn(case["out_channels"], device=device),
            training=case["training"],
        )
        ref_kwargs = dict(kwargs)
        ref_kwargs["running_mean"] = running_mean_r
        ref_kwargs["running_var"] = running_var_r
        got = candidate.relu_batch_norm_conv2d(**kwargs)
        expected = reference.relu_batch_norm_conv2d(**ref_kwargs)
        _assert_close(got, expected, atol=1e-4, rtol=1e-4)
        checks.append(f"case_{index}")
    return checks


def _case_vector_addition(candidate: Any, reference: Any, seed: int) -> list[str]:
    import torch

    del reference
    _seed(seed)
    checks: list[str] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for size in [1, 17, 1023, 1024, 1025, 4097]:
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)
        got = candidate.add(x, y)
        _assert_close(got, x + y)
        checks.append(f"size_{size}")
    return checks


def _case_adam_update_triton(candidate: Any, reference: Any, seed: int) -> list[str]:
    import torch

    _seed(seed)
    checks: list[str] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for size in [1, 31, 128, 1025]:
        p = torch.randn(size, device=device)
        grad = torch.randn(size, device=device)
        exp_avg = torch.randn(size, device=device)
        p_c, grad_c, exp_c = p.clone(), grad.clone(), exp_avg.clone()
        p_r, grad_r, exp_r = p.clone(), grad.clone(), exp_avg.clone()
        candidate.update_fn(p_c, grad_c, exp_c, 0.01, 0.01, 0.9, 0.999)
        reference.update_fn(p_r, grad_r, exp_r, 0.01, 0.01, 0.9, 0.999)
        _assert_close(p_c, p_r, atol=1e-4, rtol=1e-4)
        _assert_close(exp_c, exp_r, atol=1e-4, rtol=1e-4)
        checks.append(f"size_{size}")
    return checks


def _case_matrix_reduction(candidate: Any, reference: Any, seed: int) -> list[str]:
    del reference
    _seed(seed)
    checks: list[str] = []
    for block_m, block_n, dtype in [(1, 7, "float32"), (8, 129, "float32"), (33, 65, "float16")]:
        candidate.load_reduce(block_m, block_n, dtype)
        checks.append(f"{block_m}x{block_n}_{dtype}")
    return checks


SUITES: dict[str, Callable[[Any, Any, int], list[str]]] = {
    "relu_batch_norm_conv2d.py": _case_relu_batch_norm_conv2d,
    "vector_addition.py": _case_vector_addition,
    "adam_update_triton.py": _case_adam_update_triton,
    "matrix_reduction.py": _case_matrix_reduction,
}


def _run_one(result: dict[str, Any], seed: int) -> dict[str, Any]:
    name = result["name"]
    suite = SUITES.get(name)
    if suite is None:
        return {"supported": False, "status": "unsupported"}

    tmp_dir = Path(tempfile.mkdtemp(prefix="tritonbench_adv_"))
    try:
        candidate = _load_module_from_source(_candidate_source(Path(result["candidate"])), "candidate_mod", tmp_dir)
        reference = _load_module_from_source(_reference_source(Path(result["reference"])), "reference_mod", tmp_dir)
        checks = suite(candidate, reference, seed)
        return {"supported": True, "status": "pass", "checks": checks}
    except BaseException:
        return {"supported": True, "status": "fail", "error": traceback.format_exc()}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run hand-written adversarial rechecks for supported observable-harness pass candidates."
    )
    parser.add_argument("--observable-results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--limit", default=0, type=int)
    args = parser.parse_args()

    data = json.loads(args.observable_results.read_text(encoding="utf-8"))
    passing = [result for result in data.get("results", []) if result.get("pass")]
    if args.limit > 0:
        passing = passing[: args.limit]
    reports: list[dict[str, Any]] = []
    for result in passing:
        adversarial = _run_one(result, args.seed)
        reports.append(
            {
                "channel": result.get("channel"),
                "name": result.get("name"),
                "candidate_rel": result.get("candidate_rel"),
                "candidate": result.get("candidate"),
                "reference": result.get("reference"),
                "adversarial": adversarial,
            }
        )

    report = {
        "observable_results": str(args.observable_results),
        "pass_candidates": len(passing),
        "supported_count": sum(1 for item in reports if item["adversarial"].get("supported")),
        "adversarial_pass_count": sum(1 for item in reports if item["adversarial"].get("status") == "pass"),
        "adversarial_fail_count": sum(1 for item in reports if item["adversarial"].get("status") == "fail"),
        "results": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
