#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MARKER = "#" * 146


def _reference_path(run_dir: Path, channel: str, name: str) -> Path:
    filename = name if name.endswith(".py") else f"{name}.py"
    return run_dir / "source_data" / f"TritonBench_{channel}_v1" / filename


def _gold_test_tail(reference: Path) -> str:
    text = reference.read_text(encoding="utf-8", errors="ignore")
    if MARKER in text:
        return text.rsplit(MARKER, 1)[1].strip() + "\n"
    test_start = text.find("def test_")
    if test_start >= 0:
        return text[test_start:].strip() + "\n"
    raise ValueError(f"could not find gold test tail in {reference}")


def _has_executable_observable(reference: Path) -> bool:
    text = reference.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                return True
            if isinstance(node.func, ast.Name) and node.func.id == "assert_close":
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "assert_close":
                return True
    return False


def _load_failures(run_dir: Path) -> dict[str, set[str]]:
    summary_path = run_dir / "io_check" / "summary.json"
    if not summary_path.exists():
        return {"G": set(), "T": set()}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        channel: {Path(item["file"]).name for item in data.get("failures", [])}
        for channel, data in summary.items()
    }


def _task_names(run_dir: Path, channels: set[str], passed_only: bool) -> list[tuple[str, str]]:
    failures = _load_failures(run_dir)
    tasks: list[tuple[str, str]] = []
    for channel in ("G", "T"):
        if channel not in channels:
            continue
        root = run_dir / "source_data" / f"TritonBench_{channel}_v1"
        for reference in sorted(root.glob("*.py")):
            if passed_only and reference.name in failures.get(channel, set()):
                continue
            if _has_executable_observable(reference):
                tasks.append((channel, reference.name))
    return tasks


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

    func_name = _candidate_function_name(name)
    try:
        pattern = ast.parse(f"def {func_name}():\n    pass\n")
    except SyntaxError:
        pattern = None
    del pattern
    matches: list[Path] = []
    for path in materialized_root.rglob("*.py"):
        rel = str(path.relative_to(materialized_root))
        if channel_token not in rel:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if f"def {func_name}(" in text:
            matches.append(path)
    return sorted(matches)


def _build_script(candidate: Path, reference: Path, out: Path) -> None:
    candidate_text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
    tail = _gold_test_tail(reference)
    out.write_text(candidate_text + "\n\n" + MARKER + "\n\n" + tail, encoding="utf-8")


def _run_script(path: Path, timeout: float, seed: int) -> dict[str, Any]:
    env = None
    script = (
        "import random\n"
        f"random.seed({seed})\n"
        "try:\n"
        "    import numpy as np\n"
        f"    np.random.seed({seed})\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    import torch\n"
        f"    torch.manual_seed({seed})\n"
        "    if torch.cuda.is_available(): torch.cuda.manual_seed_all("
        f"{seed})\n"
        "except Exception:\n"
        "    pass\n"
        f"exec(compile(open({str(path)!r}, 'r', encoding='utf-8').read(), {str(path)!r}, 'exec'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(path.parent),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _progress(items: list[Any], enabled: bool) -> Any:
    if not enabled:
        return items
    try:
        from tqdm import tqdm

        return tqdm(items, desc="observable harness checks", unit="candidate")
    except ImportError:
        total = len(items)

        def _fallback() -> Any:
            for index, item in enumerate(items, 1):
                if index == 1 or index == total or index % 25 == 0:
                    print(f"observable harness checks: {index}/{total}", file=sys.stderr, flush=True)
                yield item

        return _fallback()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run generated candidates with the gold test tail for non-silent observable TritonBench tasks."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--materialized-root", type=Path)
    parser.add_argument("--channels", default="G,T", help="Comma-separated channels, e.g. G,T or T")
    parser.add_argument("--tasks", default="", help="Comma-separated task filenames/stems to include")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-candidates-per-task", default=0, type=int)
    parser.add_argument("--include-failed-gold", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    materialized_root = (
        args.materialized_root.resolve()
        if args.materialized_root
        else run_dir / "materialized_candidates"
    )
    channels = {item.strip() for item in args.channels.split(",") if item.strip()}
    task_filter = {item if item.endswith(".py") else f"{item}.py" for item in args.tasks.split(",") if item.strip()}

    tasks = _task_names(run_dir, channels, passed_only=not args.include_failed_gold)
    if task_filter:
        tasks = [(channel, name) for channel, name in tasks if name in task_filter]

    cases: list[dict[str, Any]] = []
    jobs: list[tuple[str, str, Path, Path]] = []
    for channel, name in tasks:
        reference = _reference_path(run_dir, channel, name)
        candidates = _find_candidates(materialized_root, channel, name)
        if args.max_candidates_per_task > 0:
            candidates = candidates[: args.max_candidates_per_task]
        cases.append(
            {
                "channel": channel,
                "name": name,
                "reference": str(reference),
                "candidate_count": len(candidates),
                "candidates": [str(path.relative_to(materialized_root)) for path in candidates],
            }
        )
        jobs.extend((channel, name, reference, candidate) for candidate in candidates)

    results: list[dict[str, Any]] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="tritonbench_observable_"))
    try:
        for channel, name, reference, candidate in _progress(jobs, enabled=not args.no_progress):
            safe_rel = str(candidate.relative_to(materialized_root)).replace("/", "__")
            script_path = tmp_root / channel / name / safe_rel
            script_path.parent.mkdir(parents=True, exist_ok=True)
            _build_script(candidate, reference, script_path)
            run = _run_script(script_path, args.timeout, args.seed)
            results.append(
                {
                    "channel": channel,
                    "name": name,
                    "reference": str(reference),
                    "candidate": str(candidate),
                    "candidate_rel": str(candidate.relative_to(materialized_root)),
                    "harness_script": str(script_path),
                    "pass": run["returncode"] == 0,
                    "returncode": run["returncode"],
                    "stdout": run["stdout"],
                    "stderr": run["stderr"],
                }
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    report = {
        "run_dir": str(run_dir),
        "materialized_root": str(materialized_root),
        "channels": sorted(channels),
        "tasks": task_filter and sorted(task_filter) or None,
        "case_count": len(cases),
        "candidate_total": sum(case["candidate_count"] for case in cases),
        "pass_count": sum(1 for result in results if result["pass"]),
        "fail_count": sum(1 for result in results if not result["pass"]),
        "cases": cases,
        "results": results,
    }
    text = json.dumps(report, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    summary = {key: report[key] for key in ("case_count", "candidate_total", "pass_count", "fail_count")}
    summary["out"] = str(args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
