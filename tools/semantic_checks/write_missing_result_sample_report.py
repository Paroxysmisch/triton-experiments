#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RESULT_GLOBALS = ("test_results", "result_gold", "results")


def _stdout_kind(result: dict[str, Any]) -> str:
    stdout = result.get("candidate_stdout", {}).get("stdout", "")
    if "__TRITONBENCH_RESULT_MISSING__" in stdout:
        return "missing_sentinel"
    if stdout == "":
        return "empty_crash"
    return "other"


def _is_missing_result_failure(result: dict[str, Any]) -> bool:
    return (
        result.get("strict_status") == "fail"
        and result.get("candidate_import_ok") is True
        and result.get("reference_import_ok") is True
        and result.get("candidate_result_name") is None
        and result.get("reference_result_name") is not None
    )


def _candidate_path(result: dict[str, Any], materialized_root: Path | None) -> Path:
    if materialized_root and result.get("candidate_rel"):
        return materialized_root / result["candidate_rel"]
    return Path(result["candidate"])


def _source_facts(path: Path, expected_name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text)
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    expected = next((node for node in functions if node.name == expected_name), None)
    returns: list[str] = []
    if expected:
        for node in ast.walk(expected):
            if isinstance(node, ast.Return):
                if node.value is None:
                    returns.append("bare_return")
                elif isinstance(node.value, ast.Constant) and node.value.value is None:
                    returns.append("return_none")
                else:
                    returns.append("return_value")
    assigned_globals: list[str] = []
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in RESULT_GLOBALS:
                assigned_globals.append(target.id)
    return {
        "defined_functions": [node.name for node in functions],
        "defines_expected_function": expected is not None,
        "assigned_result_globals": assigned_globals,
        "expected_function_returns": returns,
        "line_count": len(text.splitlines()),
    }


def _stderr_tail(result: dict[str, Any], limit: int = 8) -> list[str]:
    stderr = result.get("candidate_stdout", {}).get("stderr", "")
    lines = stderr.splitlines()
    return lines[-limit:]


def _explain(result: dict[str, Any], facts: dict[str, Any], kind: str) -> str:
    if kind == "missing_sentinel":
        if facts["defines_expected_function"] and "return_value" in facts["expected_function_returns"]:
            return (
                "The candidate is an implementation-only file: it defines the expected function and that "
                "function has a value-returning path, but the module never runs the benchmark test harness "
                "and never assigns test_results/result_gold/results. The appended hardener therefore reaches "
                "the end and prints __TRITONBENCH_RESULT_MISSING__."
            )
        if facts["defines_expected_function"]:
            return (
                "The candidate defines the expected function, but it still does not assign any structured "
                "result global. The failure is at the benchmark harness level: importing the file succeeds, "
                "but there is no module-level test output to compare with the reference dict."
            )
        return (
            "The candidate imports and reaches the hardener, but it does not define the expected benchmark "
            "function and does not assign any structured result global."
        )
    return (
        "The candidate imports successfully under the strict in-process importer, but when executed as a "
        "standalone script it exits with return code 1 before the appended hardener can print the missing-result "
        "sentinel. This usually means script-mode execution hits an import/decorator/top-level failure; the "
        "traceback tail below is the concrete cause for this sample."
    )


def _section(title: str, rows: list[dict[str, Any]], materialized_root: Path | None) -> list[str]:
    lines = [f"## {title}", ""]
    for index, result in enumerate(rows, 1):
        path = _candidate_path(result, materialized_root)
        facts = _source_facts(path, Path(result["name"]).stem)
        kind = _stdout_kind(result)
        lines.extend(
            [
                f"### {index}. {result['channel']} / {result['name']}",
                "",
                f"- candidate: `{result['candidate_rel']}`",
                f"- speedup: `{result.get('speedup')}`",
                f"- candidate import ok: `{result.get('candidate_import_ok')}`",
                f"- candidate script return code: `{result.get('candidate_stdout', {}).get('returncode')}`",
                f"- candidate stdout kind: `{kind}`",
                f"- reference result global: `{result.get('reference_result_name')}`",
                f"- candidate assigned result globals: `{facts['assigned_result_globals']}`",
                f"- defines expected function: `{facts['defines_expected_function']}`",
                f"- expected function returns: `{facts['expected_function_returns']}`",
                "",
                _explain(result, facts, kind),
                "",
            ]
        )
        if kind == "empty_crash":
            tail = _stderr_tail(result)
            lines.append("Traceback tail:")
            lines.append("")
            lines.append("```text")
            lines.extend(tail or ["<empty stderr>"])
            lines.append("```")
            lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a short report explaining samples of missing structured-result failures."
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--materialized-root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-bucket", default=5, type=int)
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows = [result for result in data.get("results", []) if _is_missing_result_failure(result)]
    missing = [result for result in rows if _stdout_kind(result) == "missing_sentinel"][: args.per_bucket]
    crashes = [result for result in rows if _stdout_kind(result) == "empty_crash"][: args.per_bucket]

    lines = [
        "# Missing Structured Result Sample Analysis",
        "",
        f"- input results: `{args.results}`",
        f"- total missing structured-result failures: `{len(rows)}`",
        f"- missing-sentinel samples: `{len(missing)}`",
        f"- standalone-crash samples: `{len(crashes)}`",
        "",
        "These samples explain the `NoneType != dict` failures. In this report, `NoneType` refers to the missing module-level structured result global in the candidate, not necessarily to the candidate function returning `None`.",
        "",
    ]
    lines.extend(_section("Reached Hardener But No Result Global", missing, args.materialized_root))
    lines.extend(_section("Crashed Before Hardener Printed", crashes, args.materialized_root))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
