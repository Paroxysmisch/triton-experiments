#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any


RESULT_GLOBALS = ("test_results", "result_gold", "results")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _function_returns(tree: ast.AST, function_name: str) -> list[str]:
    returns: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    if child.value is None:
                        returns.append("bare_return")
                    elif isinstance(child.value, ast.Constant) and child.value.value is None:
                        returns.append("return_none")
                    else:
                        returns.append("return_value")
            break
    return returns


def _assigned_globals(tree: ast.AST) -> set[str]:
    assigned: set[str] = set()
    for node in tree.body if isinstance(tree, ast.Module) else []:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in RESULT_GLOBALS:
                assigned.add(target.id)
    return assigned


def _defined_functions(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    } if isinstance(tree, ast.Module) else set()


def _source_facts(path: Path, expected_function: str) -> dict[str, Any]:
    text = _read_text(path)
    if not text:
        return {"path_exists": path.exists(), "parse_ok": False}
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "path_exists": True,
            "parse_ok": False,
            "syntax_error": f"{exc.msg} at line {exc.lineno}",
        }
    functions = _defined_functions(tree)
    returns = _function_returns(tree, expected_function)
    assigned = _assigned_globals(tree)
    return {
        "path_exists": True,
        "parse_ok": True,
        "defines_expected_function": expected_function in functions,
        "defined_functions": sorted(functions),
        "assigned_result_globals": sorted(assigned),
        "expected_function_returns": returns,
        "expected_function_has_explicit_return": bool(returns),
        "expected_function_returns_value": "return_value" in returns,
        "expected_function_returns_none": any(item in {"bare_return", "return_none"} for item in returns),
    }


def _is_missing_structured_result_failure(result: dict[str, Any]) -> bool:
    return (
        result.get("strict_status") == "fail"
        and result.get("candidate_import_ok") is True
        and result.get("reference_import_ok") is True
        and result.get("candidate_result_name") is None
        and result.get("reference_result_name") is not None
    )


def _candidate_stdout_kind(result: dict[str, Any]) -> str:
    stdout = result.get("candidate_stdout", {}).get("stdout", "")
    if "__TRITONBENCH_RESULT_MISSING__" in stdout:
        return "missing_sentinel"
    if "__TRITONBENCH_RESULT__=" in stdout:
        return "structured_result"
    if stdout == "":
        return "empty"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify strict failures where the candidate imported but exposed no structured result global."
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument(
        "--materialized-root",
        type=Path,
        help="Optional candidate root. When set, candidate_rel is resolved under this root instead of using absolute paths from the JSON.",
    )
    parser.add_argument("--limit", default=40, type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for result in data.get("results", []):
        if not _is_missing_structured_result_failure(result):
            continue
        candidate = (
            args.materialized_root / result["candidate_rel"]
            if args.materialized_root and result.get("candidate_rel")
            else Path(result["candidate"])
        )
        facts = _source_facts(candidate, Path(result["name"]).stem)
        stdout_kind = _candidate_stdout_kind(result)
        counters["missing_structured_result_failures"] += 1
        counters[f"stdout:{stdout_kind}"] += 1
        counters[f"parse_ok:{facts.get('parse_ok')}"] += 1
        counters[f"defines_expected_function:{facts.get('defines_expected_function')}"] += 1
        counters[f"has_result_global:{bool(facts.get('assigned_result_globals'))}"] += 1
        counters[f"returns_value:{facts.get('expected_function_returns_value')}"] += 1
        counters[f"returns_none:{facts.get('expected_function_returns_none')}"] += 1
        rows.append(
            {
                "channel": result.get("channel"),
                "name": result.get("name"),
                "speedup": result.get("speedup"),
                "candidate_rel": result.get("candidate_rel"),
                "candidate": result.get("candidate"),
                "reference_result_name": result.get("reference_result_name"),
                "candidate_import_ok": result.get("candidate_import_ok"),
                "reference_import_ok": result.get("reference_import_ok"),
                "candidate_stdout_kind": stdout_kind,
                **facts,
            }
        )

    report = {
        "results": str(args.results),
        "total_results": len(data.get("results", [])),
        "summary": dict(sorted(counters.items())),
        "examples": rows[: args.limit],
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
