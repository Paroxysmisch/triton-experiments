#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from inspect_experiment_results import inspect
from strict_io_compare import compare


def _reference_path(repo_root: Path, channel: str, name: str) -> Path:
    filename = name if name.endswith(".py") else f"{name}.py"
    return repo_root / "data" / f"TritonBench_{channel}_v1" / filename


def _candidate_function_name(name: str) -> str:
    return Path(name).stem


def _find_candidates(materialized_root: Path, channel: str, name: str) -> list[Path]:
    filename = name if name.endswith(".py") else f"{name}.py"
    channel_token = f"Bench_{channel}_"
    direct = [
        path for path in materialized_root.rglob(filename)
        if channel_token in str(path.relative_to(materialized_root))
    ]
    if direct:
        return sorted(direct)

    func_name = re.escape(_candidate_function_name(name))
    pattern = re.compile(rf"def\s+{func_name}\s*\(")
    matches: list[Path] = []
    for path in materialized_root.rglob("*.py"):
        rel = str(path.relative_to(materialized_root))
        if channel_token not in rel:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if pattern.search(text):
            matches.append(path)
    return sorted(matches)


def _case_record(case: Any, candidates: list[Path], repo_root: Path) -> dict[str, Any]:
    return {
        "channel": case.channel,
        "name": case.name,
        "speedup": case.speedup,
        "reference": str(_reference_path(repo_root, case.channel, case.name)),
        "candidate_count": len(candidates),
        "candidates": [str(path) for path in candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict I/O checks on L40S weak-oracle speedup candidates.")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument(
        "--selection-repo-root",
        type=Path,
        help=(
            "Repository root used only for selecting weak-oracle cases. "
            "This should point at the original sources when --repo-root points "
            "at hardened reference files."
        ),
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--materialized-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-candidates-per-case", default=0, type=int)
    parser.add_argument(
        "--print-full-report",
        action="store_true",
        help="Print the complete JSON report to stdout. By default, --out gets the full report and stdout gets a compact summary.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    selection_repo_root = (
        args.selection_repo_root.resolve()
        if args.selection_repo_root
        else repo_root
    )
    run_dir = args.run_dir.resolve()
    materialized_root = (
        args.materialized_root.resolve()
        if args.materialized_root
        else run_dir / "materialized_candidates"
    )

    cases = [
        case for case in inspect(selection_repo_root, run_dir)
        if not case.io_failed and case.speedup > 1.0 and case.weak_stdout_oracle
    ]
    report: dict[str, Any] = {
        "repo_root": str(repo_root),
        "selection_repo_root": str(selection_repo_root),
        "run_dir": str(run_dir),
        "materialized_root": str(materialized_root),
        "seed": args.seed,
        "timeout": args.timeout,
        "dry_run": args.dry_run,
        "case_count": len(cases),
        "cases": [],
        "results": [],
    }

    for case in cases:
        candidates = _find_candidates(materialized_root, case.channel, case.name)
        if args.max_candidates_per_case > 0:
            candidates = candidates[: args.max_candidates_per_case]
        report["cases"].append(_case_record(case, candidates, repo_root))
        if args.dry_run:
            continue
        reference = _reference_path(repo_root, case.channel, case.name)
        for candidate in candidates:
            result = compare(candidate, reference, args.timeout, args.seed)
            result.update(
                {
                    "channel": case.channel,
                    "name": case.name,
                    "speedup": case.speedup,
                    "candidate_rel": str(candidate.relative_to(materialized_root)),
                }
            )
            report["results"].append(result)

    if not args.dry_run:
        report["stdout_false_positive_count"] = sum(
            1 for result in report["results"] if result.get("stdout_false_positive")
        )
        report["strict_fail_count"] = sum(
            1 for result in report["results"] if result.get("strict_status") == "fail"
        )
        report["import_error_count"] = sum(
            1 for result in report["results"] if result.get("strict_status") == "import_error"
        )

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    summary = {
        "repo_root": report["repo_root"],
        "selection_repo_root": report["selection_repo_root"],
        "run_dir": report["run_dir"],
        "materialized_root": report["materialized_root"],
        "seed": report["seed"],
        "timeout": report["timeout"],
        "dry_run": report["dry_run"],
        "case_count": report["case_count"],
        "candidate_total": sum(case["candidate_count"] for case in report["cases"]),
        "result_count": len(report["results"]),
        "stdout_false_positive_count": report.get("stdout_false_positive_count"),
        "strict_fail_count": report.get("strict_fail_count"),
        "import_error_count": report.get("import_error_count"),
        "strict_pass_count": sum(
            1 for result in report["results"] if result.get("strict_status") == "pass"
        ),
        "out": str(args.out) if args.out else None,
    }
    print(text if args.print_full_report else json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
