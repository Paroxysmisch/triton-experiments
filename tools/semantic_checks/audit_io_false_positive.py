#!/usr/bin/env python3
"""Audit TritonBench-T generated solutions for stdout-IO false positives.

This script is intentionally small and conservative.  It does two different
things:

1. Reproduces the original TritonBench-T execution-accuracy classifier:
   generated stdout == golden stdout.  The original script does not inspect
   returned tensors in Python; it only compares captured stdout.
2. Runs a handful of deterministic semantic checkers for common generated-code
   bugs, producing concrete counterexamples where the PyTorch contract and the
   generated solution disagree.

The semantic checkers do not require CUDA, torch, or triton.  They are not a
formal verifier; they are a triage tool for finding examples worth feeding into
an SMT/verification pipeline.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
HF_T_JSON = REPO_ROOT / "hf_datasets" / "tritonbench_t" / "TritonBench_T_v1.json"
GOLD_T_DIR = REPO_ROOT / "data" / "TritonBench_T_v1"
SEPARATOR = "#" * 146


@dataclass
class Counterexample:
    kind: str
    explanation: str
    inputs: dict[str, Any]
    expected: Any
    observed: Any


@dataclass
class AuditReport:
    jsonl: str
    line: int
    task_file: str
    task_name: str
    extracted_path: str | None
    gold_stdout_empty_static: bool
    generated_stdout_empty_static: bool
    static_stdout_classifier_would_accept_if_both_exit: bool
    runtime_stdout_classifier: dict[str, Any] | None
    counterexamples: list[Counterexample]
    notes: list[str]


def load_hf_t() -> list[dict[str, Any]]:
    with HF_T_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_line(path: Path, line: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            if idx == line:
                return json.loads(raw)
    raise ValueError(f"{path} has fewer than {line} lines")


def clear_code(text: str) -> str:
    if "```python" in text:
        text = text.split("```python", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    return (
        text.replace("<|im_end|>", "")
        .replace("<|EOT|>", "")
        .replace("</s>", "")
        .strip()
    )


def infer_task_from_line(line: int, item: dict[str, Any], task_override: str | None) -> dict[str, Any]:
    hf = load_hf_t()
    if task_override:
        for task in hf:
            if task["file"] == task_override or task.get("name") == task_override:
                return task
        raise ValueError(f"Could not find task {task_override!r} in {HF_T_JSON}")

    # Most checked-in T JSONLs preserve the HF task order.
    if 1 <= line <= len(hf):
        return hf[line - 1]

    # Fallback: match description embedded in the instruction/prompt.
    prompt = item.get("instruction") or item.get("prompt") or ""
    if "Functional Description:" in prompt and "Wrapper Entry Information:" in prompt:
        desc = prompt.split("Functional Description:", 1)[1].split(
            "Wrapper Entry Information:", 1
        )[0]
        desc = " ".join(desc.split())
        matches = [
            task
            for task in hf
            if desc and desc in " ".join(task.get("description", "").split())
        ]
        if len(matches) == 1:
            return matches[0]

    raise ValueError("Could not infer task; pass --task-file")


def split_gold_test(task_file: str) -> str:
    path = GOLD_T_DIR / task_file
    text = path.read_text(encoding="utf-8")
    if SEPARATOR not in text:
        raise ValueError(f"Cannot find TritonBench separator in {path}")
    return text.split(SEPARATOR, 1)[1]


def has_top_level_print(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bool(re.search(r"(^|\n)\s*print\s*\(", code))
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "print":
                return True
    return False


def run_script(path: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def reproduce_stdout_classifier(
    generated_code: str,
    gold_test: str,
    gold_file: Path,
    timeout: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tritonbench_io_audit_") as tmp:
        generated_path = Path(tmp) / "generated.py"
        generated_path.write_text(
            generated_code + "\n" + SEPARATOR + "\n" + gold_test,
            encoding="utf-8",
        )
        gen = run_script(generated_path, timeout)
        gold = run_script(gold_file, timeout)
    return {
        "generated_returncode": gen.returncode,
        "gold_returncode": gold.returncode,
        "generated_stdout": gen.stdout,
        "gold_stdout": gold.stdout,
        "generated_stderr_excerpt": gen.stderr[-1000:],
        "gold_stderr_excerpt": gold.stderr[-1000:],
        "eval_T_1_exe_acc_would_accept": gen.stdout == gold.stdout,
        "note": "This exactly mirrors EVAL/eval_T/1_exe_acc.py stdout comparison; return codes are shown but not part of that classifier.",
    }


def add_counterexamples(task_file: str, code: str) -> list[Counterexample]:
    if task_file != "add.py":
        return []
    lowered = re.sub(r"\s+", "", code)
    alpha_is_in_signature = "alpha" in code
    likely_naive_add = any(pat in lowered for pat in ["x+y", "input+other", "a+b"])
    alpha_used_in_arithmetic = bool(re.search(r"alpha\s*\*", code) or re.search(r"\*\s*alpha", code))
    scalar_branch = "isinstance(other" in code or "isinstance(y" in code
    examples: list[Counterexample] = []
    if likely_naive_add and (not alpha_used_in_arithmetic):
        examples.append(
            Counterexample(
                kind="add_alpha_ignored",
                explanation="PyTorch add computes input + alpha * other; generated code appears to compute input + other.",
                inputs={"input": [0.0], "other": [1.0], "alpha": 2.0},
                expected=[2.0],
                observed=[1.0],
            )
        )
    if not scalar_branch:
        examples.append(
            Counterexample(
                kind="add_scalar_other_unsupported",
                explanation="PyTorch add accepts scalar `other`; generated code appears to require tensor pointers.",
                inputs={"input": [3.0], "other": 2.0, "alpha": 0.5},
                expected=[4.0],
                observed="unsupported or wrong without scalar branch",
            )
        )
    if alpha_is_in_signature and not alpha_used_in_arithmetic:
        examples.append(
            Counterexample(
                kind="add_default_tests_can_miss_alpha",
                explanation="If tests use only alpha=1, ignoring alpha is invisible to input-output tests.",
                inputs={"test_condition": "alpha == 1"},
                expected="input + other",
                observed="input + other",
            )
        )
    return examples


def div_counterexamples(task_file: str, code: str) -> list[Counterexample]:
    if task_file != "div.py":
        return []
    examples: list[Counterexample] = []
    rounding_mentions = "rounding_mode" in code
    trunc_or_floor_used = any(tok in code for tok in ["tl.floor", "torch.floor", "trunc", "tl.math.floor"])
    scalar_branch = "isinstance(other" in code or "isinstance(y" in code
    broadcast_branch = "broadcast" in code or ".expand(" in code

    if (not rounding_mentions) or (rounding_mentions and not trunc_or_floor_used):
        examples.append(
            Counterexample(
                kind="div_rounding_mode_ignored",
                explanation="PyTorch div supports rounding_mode='floor'/'trunc'; generated code appears to perform true division only.",
                inputs={"input": [-3], "other": [2], "rounding_mode": "floor"},
                expected=[-2],
                observed=[-1.5],
            )
        )
    if not scalar_branch:
        examples.append(
            Counterexample(
                kind="div_scalar_other_unsupported",
                explanation="PyTorch div accepts scalar divisors; generated code appears to require tensor pointers.",
                inputs={"input": [4.0], "other": 2.0, "rounding_mode": None},
                expected=[2.0],
                observed="unsupported or wrong without scalar branch",
            )
        )
    if not broadcast_branch:
        examples.append(
            Counterexample(
                kind="div_broadcast_unsupported",
                explanation="PyTorch div broadcasts input and other; generated code appears same-shape only.",
                inputs={"input_shape": [2, 2], "other_shape": [2]},
                expected="shape [2, 2]",
                observed="same-shape-only pointer indexing",
            )
        )
    return examples


def broadcast_counterexamples(task_file: str, code: str) -> list[Counterexample]:
    if task_file != "broadcast_tensors.py":
        return []
    lowered = re.sub(r"\s+", "", code)
    examples: list[Counterexample] = []
    if "return[output,output]" in lowered or "return(output,output)" in lowered:
        examples.append(
            Counterexample(
                kind="broadcast_returns_duplicate_sum",
                explanation="broadcast_tensors should return broadcasted input views/values, not two copies of an addition output.",
                inputs={"x": [[0, 1, 2]], "y": [[0], [1]]},
                expected=(
                    "a=[[0,1,2],[0,1,2]], "
                    "b=[[0,0,0],[1,1,1]]"
                ),
                observed="two identical output tensors",
            )
        )
    if "zip(" in code and "reversed" not in code and "broadcast" not in code:
        examples.append(
            Counterexample(
                kind="broadcast_left_aligned",
                explanation="PyTorch broadcasting aligns dimensions from the right; generated code appears to compare shapes left-to-right.",
                inputs={"x_shape": [3], "y_shape": [2, 1]},
                expected=[2, 3],
                observed="likely rejected or mis-shaped",
            )
        )
    return examples


def std_counterexamples(task_file: str, code: str) -> list[Counterexample]:
    if task_file != "std.py":
        return []
    examples: list[Counterexample] = []
    if ".numel()" in code and ("dim" in code):
        examples.append(
            Counterexample(
                kind="std_dim_uses_global_numel",
                explanation="torch.std(input, dim=...) reduces over the selected dimension; generated code appears to use global numel.",
                inputs={"input": [[1.0, 3.0], [5.0, 7.0]], "dim": 1, "correction": 0},
                expected=[1.0, 1.0],
                observed="would divide/reduce using all 4 elements or otherwise global size",
            )
        )
    if "abs(" in code and "sqrt" not in code:
        examples.append(
            Counterexample(
                kind="std_mean_absolute_deviation",
                explanation="Standard deviation is sqrt(mean squared deviation); generated fallback appears to use absolute deviation.",
                inputs={"input": [0.0, 0.0, 2.0], "correction": 0},
                expected=[0.9428090416],
                observed=[0.8888888889],
            )
        )
    return examples


def gelu_counterexamples(task_file: str, code: str) -> list[Counterexample]:
    examples: list[Counterexample] = []
    if task_file == "add_gelu.py":
        alpha_used = bool(re.search(r"alpha\s*\*", code) or re.search(r"\*\s*alpha", code))
        if not alpha_used:
            examples.append(
                Counterexample(
                    kind="add_gelu_alpha_ignored",
                    explanation="Spec is gelu(input + alpha * other); generated code appears to omit alpha.",
                    inputs={"input": [0.0], "other": [1.0], "alpha": 2.0},
                    expected="gelu(2.0)",
                    observed="gelu(1.0)",
                )
            )
    if task_file == "sub_gelu.py":
        suspicious = "input * other" in code or "x * other" in code or "other * other" in code
        if suspicious:
            examples.append(
                Counterexample(
                    kind="sub_gelu_wrong_formula",
                    explanation="Spec is gelu(input - alpha * other); generated code appears to multiply input/other or cube the wrong term.",
                    inputs={"input": [3.0], "other": [1.0], "alpha": 2.0},
                    expected="gelu(1.0)",
                    observed="formula over input*other or wrong cubic term",
                )
            )
    return examples


def find_counterexamples(task_file: str, code: str) -> list[Counterexample]:
    checks = [
        add_counterexamples,
        div_counterexamples,
        broadcast_counterexamples,
        std_counterexamples,
        gelu_counterexamples,
    ]
    out: list[Counterexample] = []
    for check in checks:
        out.extend(check(task_file, code))
    return out


def write_extracted(
    generated_code: str,
    gold_test: str,
    out_dir: Path | None,
    task_file: str,
) -> str | None:
    if out_dir is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / task_file
    path.write_text(generated_code + "\n" + SEPARATOR + "\n" + gold_test, encoding="utf-8")
    return str(path)


def audit(args: argparse.Namespace) -> AuditReport:
    jsonl_path = Path(args.jsonl).resolve()
    item = load_jsonl_line(jsonl_path, args.line)
    task = infer_task_from_line(args.line, item, args.task_file)
    task_file = task["file"]
    task_name = task.get("name", task_file)
    generated_code = clear_code(item.get("predict", ""))
    gold_test = split_gold_test(task_file)
    extracted_path = write_extracted(generated_code, gold_test, args.extract_dir, task_file)

    gold_stdout_empty_static = not has_top_level_print(gold_test)
    generated_stdout_empty_static = not has_top_level_print(generated_code)
    notes = [
        "TritonBench-T generated JSONLs usually follow HF dataset line order; pass --task-file to override.",
        "Static stdout analysis is conservative: nested or imported prints may evade it.",
    ]

    runtime = None
    if args.run_stdout_classifier:
        runtime = reproduce_stdout_classifier(
            generated_code=generated_code,
            gold_test=gold_test,
            gold_file=GOLD_T_DIR / task_file,
            timeout=args.timeout,
        )

    return AuditReport(
        jsonl=str(jsonl_path),
        line=args.line,
        task_file=task_file,
        task_name=task_name,
        extracted_path=extracted_path,
        gold_stdout_empty_static=gold_stdout_empty_static,
        generated_stdout_empty_static=generated_stdout_empty_static,
        static_stdout_classifier_would_accept_if_both_exit=(
            gold_stdout_empty_static and generated_stdout_empty_static
        ),
        runtime_stdout_classifier=runtime,
        counterexamples=find_counterexamples(task_file, generated_code),
        notes=notes,
    )


def to_jsonable(report: AuditReport) -> dict[str, Any]:
    data = asdict(report)
    data["counterexamples"] = [asdict(c) for c in report.counterexamples]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a TritonBench-T generated solution and audit stdout IO false positives."
    )
    parser.add_argument("--jsonl", required=True, help="Path to an LLM_generated JSONL file")
    parser.add_argument("--line", required=True, type=int, help="1-indexed JSONL line/task number")
    parser.add_argument("--task-file", help="Override inferred HF task file, e.g. add.py")
    parser.add_argument("--extract-dir", type=Path, help="Optional directory to write extracted .py solution")
    parser.add_argument(
        "--run-stdout-classifier",
        action="store_true",
        help="Execute generated and gold scripts and compare stdout like EVAL/eval_T/1_exe_acc.py",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    report = audit(args)
    print(json.dumps(to_jsonable(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
