#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CASE_NOTES: dict[str, dict[str, str]] = {
    "fused_recurrent_retention": {
        "why_wrong": "Several generated implementations use the wrong recurrence decay or the wrong public API. The gold kernel uses `1 - 2^(-5 - head)`, while representative generated code uses `1 / 2^(head + 1)` or renames/misses `output_final_state`.",
        "counterexample": "Call `fused_recurrent_retention(q, k, v, initial_state=None, output_final_state=True)` and compare `(output, final_state)`. For head 0, a decay of `0.5` instead of `0.96875` causes a large output/final-state mismatch.",
    },
    "l2_norm_bwd": {
        "why_wrong": "Representative false-positive candidates expose the wrong function name/signature. The gold API is `_l2_norm_bwd(x, dy, eps=1e-5)`, while one candidate defines `l2_norm_bwd(X, DY, variance, rstd, BLOCK_SIZE=256)`. Other generated variants reverse `(x, dy)` or use a different default epsilon.",
        "counterexample": "Call `_l2_norm_bwd(torch.randn(1, 8, device='cuda'), torch.randn(1, 8, device='cuda'))`; candidates without `_l2_norm_bwd` fail immediately. For epsilon bugs, use a near-zero row such as `x = full((2, 8), 1e-8)`.",
    },
    "triton_mul2": {
        "why_wrong": "Representative false-positive candidates expose the wrong public API. The gold function is `triton_mul2(x, BLOCK_SIZE=16)` and returns a tensor. A generated candidate defines `triton_mul2(x, y)`, requires caller-provided output, and returns nothing.",
        "counterexample": "Call `triton_mul2(torch.tensor([1., -3.], device='cuda'), BLOCK_SIZE=2)` and expect `[2., -6.]`. The representative candidate raises a signature error or returns `None`.",
    },
    "broadcast_tensors": {
        "why_wrong": "The gold is `torch.broadcast_tensors(x, y)`. A representative false-positive candidate calls `tl.broadcast_tensors` without importing `tl`; Triton language also has no Python-level replacement with PyTorch's view/stride semantics.",
        "counterexample": "Use `x = torch.tensor(3.0, device='cuda')`, `y = torch.tensor([1.,2.,3.], device='cuda')`; the gold returns two broadcasted tensors, while the representative candidate raises `NameError` or lacks the API.",
    },
    "tensordot_rsqrt": {
        "why_wrong": "The gold computes `torch.rsqrt(torch.tensordot(a, b, dims=dims))`. A representative candidate defines a `@triton.jit` function as if it were a normal Python function, uses tensor objects directly inside JIT, and passes `dims` as a Triton reduction axis.",
        "counterexample": "Use `a=[1,2,3]`, `b=[4,5,6]`, `dims=1`; the gold is `rsqrt(32)`, while the representative JIT-style candidate is not legally callable as this Python wrapper.",
    },
    "sum_std": {
        "why_wrong": "The gold computes `summed = input.sum(...)` and then `std` over that result. A representative candidate reshapes the scalar std to `[1] * input.ndim` when `keepdim=True`; for larger reductions it also stores only `pid == 0`, ignoring later blocks.",
        "counterexample": "Use `x = tensor([[1.,2.],[3.,4.]], device='cuda')` with `dim=0, keepdim=True`. The gold std is scalar-shaped, while the representative candidate returns shape `(1, 1)`.",
    },
    "sigmoid_adaptive_avg_pool2d": {
        "why_wrong": "The gold computes `torch.sigmoid(F.adaptive_avg_pool2d(input, output_size))`. A representative generated Triton version passes `input.data_ptr()` integers into a kernel launch and uses Python `range(h_start, h_end)` where `h_start/h_end` are Triton values, so the kernel is not a valid implementation.",
        "counterexample": "Use `x = torch.randn(1, 3, 8, 8, device='cuda')`, `output_size=4`. The gold returns shape `(1,3,4,4)`; the representative generated code fails to launch/compile correctly.",
    },
    "fused_cross_entropy_log_softmax": {
        "why_wrong": "The gold combines cross entropy and log softmax with PyTorch semantics, including weights, ignore_index, and reduction. Generated implementations frequently divide weighted mean losses by the wrong denominator.",
        "counterexample": "Use class weights, `ignore_index`, and `reduction='mean'`; the mean must divide by the sum of selected target weights, not merely by the count.",
    },
    "fused_qr_solve": {
        "why_wrong": "The gold solves a linear system via QR and returns a numerical solution. Several generated files are incomplete, use placeholders, or do not expose a result dictionary under the strict checker.",
        "counterexample": "Use a full-rank rectangular `A` and `b`, then check the residual `norm(A @ x - b)` against the gold solution.",
    },
    "fused_transformer_block": {
        "why_wrong": "The gold block includes linear/normalization/dropout-style behavior with specific argument semantics. Generated versions often simplify stochastic dropout or partial block behavior, which stdout cannot observe.",
        "counterexample": "Set `training=False` and `p=0` to remove randomness, compare deterministic output; then test seeded dropout separately.",
    },
    "dropout_sigmoid_linear": {
        "why_wrong": "Dropout is stochastic in training and identity in eval. Generated candidates often hard-code or mishandle dropout masks/scaling, and stdout equality cannot observe returned tensors.",
        "counterexample": "Check `training=False` and `p=0` first, then seeded `training=True, p=0.5` with PyTorch's expected dropout scaling.",
    },
    "elu_linear": {
        "why_wrong": "The gold applies a linear operation and ELU with PyTorch's shape, bias, alpha, and optional output semantics. Generated candidates often flatten or specialize the matrix case.",
        "counterexample": "Use `bias=None`, non-default `alpha`, and a non-contiguous input; compare to the PyTorch reference exactly.",
    },
    "det": {
        "why_wrong": "The gold is determinant with PyTorch's batched and singular-matrix semantics. Generated code may compute via a restricted LU/log-det path and miss signs, batches, or singular behavior.",
        "counterexample": "Use batched matrices including a negative determinant and a singular matrix; compare `torch.det` results including zero/signed behavior.",
    },
    "lu.py": {
        "why_wrong": "The gold exposes PyTorch LU behavior including pivoting modes and returned pivots/info. Generated wrappers often only cover the simplest pivoted case.",
        "counterexample": "Exercise `pivot=False` when supported and compare the full tuple, including pivots/info, not just the reconstructed matrix.",
    },
    "invert_matrix_lu": {
        "why_wrong": "The gold computes matrix inverse through LU semantics. Generated files can pass empty stdout while not proving correct inverses, pivot handling, or batched behavior.",
        "counterexample": "Use a batched well-conditioned matrix and verify both `A @ inv(A)` and `inv(A) @ A` are identity within tolerance.",
    },
}


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("`", "\\`")


def _code(text: str, limit: int = 80) -> str:
    lines = text.strip("\n").splitlines()
    if len(lines) > limit:
        lines = lines[:limit] + [f"... <truncated; {len(text.splitlines()) - limit} more lines>"]
    return "```python\n" + "\n".join(lines) + "\n```"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _resolve_path(raw: str, repo_root: Path, run_dir: Path, materialized_root: Path) -> Path:
    path = Path(raw)
    parts = path.parts
    if "materialized_candidates" in parts:
        idx = parts.index("materialized_candidates")
        return materialized_root.joinpath(*parts[idx + 1 :])
    if "data" in parts:
        idx = parts.index("data")
        rel = Path(*parts[idx + 1 :])
        source_candidate = run_dir / "source_data" / rel
        if source_candidate.exists():
            return source_candidate
        return repo_root / "data" / rel
    return path


def _source_path(case: dict[str, Any], repo_root: Path, run_dir: Path) -> Path:
    name = case["name"]
    filename = name if name.endswith(".py") else f"{name}.py"
    rel = Path(f"TritonBench_{case['channel']}_v1") / filename
    source_candidate = run_dir / "source_data" / rel
    if source_candidate.exists():
        return source_candidate
    return repo_root / "data" / rel


def _split_gold(source: str) -> tuple[str, str]:
    marker = "#" * 146
    if marker in source:
        before, after = source.rsplit(marker, 1)
        return before.strip(), after.strip()
    test_idx = source.find("def test_")
    if test_idx >= 0:
        return source[:test_idx].strip(), source[test_idx:].strip()
    return source.strip(), ""


def _representative(results: list[dict[str, Any]], materialized_root: Path) -> dict[str, Any]:
    def score(result: dict[str, Any]) -> tuple[int, int]:
        rel = result.get("candidate_rel", "")
        path = materialized_root / rel
        text = _read(path) if path.exists() else ""
        semantic_markers = (
            "pass",
            "TODO",
            "NotImplemented",
            "return None",
            "gamma =",
            "eps=1e-12",
            "eps: float",
            "torch.",
            "triton.jit",
        )
        marker_score = sum(marker in text for marker in semantic_markers)
        return (marker_score, -len(text))

    return sorted(results, key=score, reverse=True)[0]


def _summary_table(rows: list[tuple[str, str]]) -> str:
    body = []
    for left, right in rows:
        body.append(f"  [{left}], [{right}],")
    return "#table(\n  columns: 2,\n  [field], [value],\n" + "\n".join(body) + "\n)\n"


def build_report(data: dict[str, Any], repo_root: Path, run_dir: Path, materialized_root: Path) -> str:
    cases_by_key = {(c["channel"], c["name"]): c for c in data["cases"]}
    fp_by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in data["results"]:
        if result.get("stdout_false_positive"):
            fp_by_case[(result["channel"], result["name"])].append(result)

    out: list[str] = []
    out.append("= Strict I/O False Positives in TritonBench L40S\n")
    out.append("#set text(size: 9pt)\n#set heading(numbering: \"1.\")\n#set page(margin: 0.75in)\n")
    out.append("This report is generated from the clean full strict I/O rerun. It focuses on cases where the repository-style stdout oracle accepts the generated file, but the stricter checker finds missing or mismatched structured outputs.\n")
    out.append(_summary_table([
        ("run directory", f"`{_escape(str(run_dir))}`"),
        ("weak-oracle cases", f"`{data.get('case_count')}`"),
        ("matched candidates", f"`{len(data.get('results', []))}`"),
        ("stdout false positives", f"`{data.get('stdout_false_positive_count')}`"),
        ("strict failures", f"`{data.get('strict_fail_count')}`"),
        ("import errors", f"`{data.get('import_error_count')}`"),
    ]))
    out.append("== How To Read The Evidence\n")
    out.append("For each task, TritonBench has a gold source file whose test tail builds `result_gold`, `results`, or `test_results`. The weak I/O oracle compares only captured stdout. Many gold tests do not print the result dictionary, and many materialized generated files also print nothing. Therefore `stdout == stdout` can be true even when the generated implementation returns nothing, exposes no result global, or computes different tensors.\n")
    out.append("The strict rerun imports the generated file and the gold file with a fixed seed, normalizes result globals, and compares them. The common mismatch `$: type differs: NoneType != dict` means the candidate exposed no structured result while the gold test produced a dictionary.\n")

    for key in sorted(fp_by_case):
        channel, name = key
        case = cases_by_key[key]
        results = fp_by_case[key]
        rep = _representative(results, materialized_root)
        source_path = _source_path(case, repo_root, run_dir)
        candidate_path = materialized_root / rep["candidate_rel"]
        source_text = _read(source_path) if source_path.exists() else ""
        gold_impl, gold_tests = _split_gold(source_text)
        candidate_text = _read(candidate_path) if candidate_path.exists() else ""
        note = CASE_NOTES.get(name, {})

        out.append(f"== `{channel}` / `{name}`\n")
        out.append(_summary_table([
            ("speedup", f"`{case.get('speedup')}`"),
            ("stdout false-positive candidates", f"`{len(results)}`"),
            ("representative candidate", f"`{_escape(rep['candidate_rel'])}`"),
            ("strict mismatch", f"`{_escape('; '.join(rep.get('strict_mismatches', [])[:3]))}`"),
            ("candidate stdout", f"`{_escape(repr(rep.get('candidate_stdout', {}).get('stdout', '')))}`"),
            ("reference stdout", f"`{_escape(repr(rep.get('reference_stdout', {}).get('stdout', '')))}`"),
        ]))
        out.append("=== GT Implementation Snippet\n")
        out.append(_code(gold_impl, 70) + "\n")
        out.append("=== Generated Solution Snippet\n")
        out.append(_code(candidate_text, 90) + "\n")
        out.append("=== I/O Test Inputs That Are Treated As Passing\n")
        out.append("The weak oracle sees this as passing because the generated script and the gold script have identical stdout. The recovered gold test tail below shows the inputs and result dictionary that should have been compared structurally.\n")
        out.append(_code(gold_tests, 90) + "\n")
        out.append("=== Why The I/O Test Passes\n")
        out.append(f"- candidate stdout: `{_escape(repr(rep.get('candidate_stdout', {}).get('stdout', '')))}`\n")
        out.append(f"- reference stdout: `{_escape(repr(rep.get('reference_stdout', {}).get('stdout', '')))}`\n")
        out.append("- the stdout strings are equal, so the weak oracle accepts the generated file;\n")
        out.append(f"- strict comparison then reports: `{_escape('; '.join(rep.get('strict_mismatches', [])[:5]))}`.\n")
        out.append("=== Why It Is Not Correct\n")
        out.append(note.get("why_wrong", "The generated file is a strict false positive: it matches stdout but does not expose the same structured result as the gold test. This is enough to reject it under a result-aware oracle, but the exact semantic bug should be confirmed with a candidate-specific appended-test rerun.") + "\n")
        out.append("=== Obvious Counterexample / Follow-up Test\n")
        out.append(note.get("counterexample", "Append the gold test tail to the generated implementation, capture its result dictionary, and compare every tensor/value against the gold result rather than comparing stdout.") + "\n")
        out.append("=== All Stdout-False-Positive Candidates For This Task\n")
        for result in sorted(results, key=lambda r: r["candidate_rel"]):
            out.append(f"- `{_escape(result['candidate_rel'])}`: `{_escape('; '.join(result.get('strict_mismatches', [])[:2]))}`\n")

    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Typst report for strict I/O stdout false positives.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--materialized-root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    repo_root = args.repo_root.resolve()
    materialized_root = (
        args.materialized_root.resolve()
        if args.materialized_root
        else run_dir / "materialized_candidates"
    )
    report = build_report(data, repo_root, run_dir, materialized_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
