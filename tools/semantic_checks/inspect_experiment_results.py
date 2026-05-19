#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PerfCase:
    channel: str
    name: str
    speedup: float
    io_failed: bool
    source_path: Path | None
    print_count: int
    assert_count: int
    assert_close_count: int
    assigns_test_results: bool
    assigns_result_gold: bool

    @property
    def weak_stdout_oracle(self) -> bool:
        return (
            self.print_count == 0
            and self.assert_count == 0
            and (self.assigns_test_results or self.assigns_result_gold)
        )


def _load_failures(run_dir: Path) -> dict[str, set[str]]:
    summary_path = run_dir / "io_check" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failures: dict[str, set[str]] = {}
    for channel, data in summary.items():
        failures[channel] = {
            Path(item["file"]).stem
            for item in data.get("failures", [])
        }
    return failures


def _load_speedups(run_dir: Path, channel: str) -> dict[str, float]:
    summary_path = run_dir / f"perf_{channel}_summary.txt"
    text = summary_path.read_text(encoding="utf-8")
    speedups: dict[str, float] = {}
    for match in re.finditer(r"([^\s:]+\.json): ([0-9]+(?:\.[0-9]+)?)", text):
        speedups.setdefault(Path(match.group(1)).stem, float(match.group(2)))
    return speedups


def _source_path(repo_root: Path, channel: str, name: str) -> Path | None:
    data_dir = repo_root / "data" / f"TritonBench_{channel}_v1"
    candidates = [data_dir / f"{name}.py"]
    if channel == "T" and name.endswith(".py"):
        candidates.append(data_dir / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _case(repo_root: Path, run_dir: Path, channel: str, name: str, speedup: float, failures: set[str]) -> PerfCase:
    source_path = _source_path(repo_root, channel, name)
    source = source_path.read_text(encoding="utf-8", errors="ignore") if source_path else ""
    return PerfCase(
        channel=channel,
        name=name,
        speedup=speedup,
        io_failed=name in failures or f"{name}.py" in failures,
        source_path=source_path,
        print_count=source.count("print("),
        assert_count=source.count("assert"),
        assert_close_count=source.count("assert_close"),
        assigns_test_results="test_results =" in source,
        assigns_result_gold="result_gold =" in source,
    )


def inspect(repo_root: Path, run_dir: Path) -> list[PerfCase]:
    failures = _load_failures(run_dir)
    cases: list[PerfCase] = []
    for channel in ("G", "T"):
        for name, speedup in _load_speedups(run_dir, channel).items():
            cases.append(_case(repo_root, run_dir, channel, name, speedup, failures.get(channel, set())))
    return cases


def _print_cases(title: str, cases: list[PerfCase], limit: int) -> None:
    print(title)
    for case in cases[:limit]:
        source = str(case.source_path) if case.source_path else "<missing>"
        print(
            f"{case.channel:1s} {case.name:45s} "
            f"speedup={case.speedup:7.4f} "
            f"io={'FAIL' if case.io_failed else 'pass':4s} "
            f"weak_stdout={str(case.weak_stdout_oracle):5s} "
            f"print={case.print_count} assert={case.assert_count} "
            f"source={source}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect TritonBench experiment pass/fail and false-positive risk.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    cases = inspect(repo_root, run_dir)
    pass_speedups = [case for case in cases if not case.io_failed and case.speedup > 1.0]
    weak_pass_speedups = [case for case in pass_speedups if case.weak_stdout_oracle]
    failures = [case for case in cases if case.io_failed]

    print(f"cases with perf entries: {len(cases)}")
    print(f"I/O failures with perf entries: {len(failures)}")
    print(f"I/O-passing cases with speedup > 1: {len(pass_speedups)}")
    print(f"I/O-passing speedups with weak stdout oracle: {len(weak_pass_speedups)}")
    print()

    _print_cases("Top I/O-passing speedups", sorted(pass_speedups, key=lambda c: c.speedup, reverse=True), args.limit)
    _print_cases("Top weak-oracle I/O-passing speedups", sorted(weak_pass_speedups, key=lambda c: c.speedup, reverse=True), args.limit)
    _print_cases("I/O failures present in perf summaries", sorted(failures, key=lambda c: (c.channel, c.name)), args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
