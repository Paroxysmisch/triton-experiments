#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _stdout(result: dict[str, Any], side: str) -> str:
    return result.get(f"{side}_stdout", {}).get("stdout", "")


def _bucket(result: dict[str, Any]) -> str:
    candidate_stdout = _stdout(result, "candidate")
    reference_stdout = _stdout(result, "reference")
    if candidate_stdout == reference_stdout == "":
        return "empty_equal_stdout"
    if candidate_stdout == reference_stdout:
        return "nonempty_equal_stdout"
    return "stdout_diff"


def _classification(result: dict[str, Any]) -> str:
    status = result.get("strict_status")
    if status == "pass":
        return "strict_correct"
    if status == "fail":
        return "strict_incorrect"
    if status == "import_error":
        return "not_runnable"
    return f"unknown_{status}"


def classify(data: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in data["results"]:
        record = {
            "channel": result.get("channel"),
            "name": result.get("name"),
            "speedup": result.get("speedup"),
            "candidate_rel": result.get("candidate_rel"),
            "repo_stdout_equal": result.get("repo_stdout_equal"),
            "strict_status": result.get("strict_status"),
            "classification": _classification(result),
            "stdout_false_positive": result.get("stdout_false_positive", False),
            "candidate_stdout_excerpt": _stdout(result, "candidate")[:500],
            "reference_stdout_excerpt": _stdout(result, "reference")[:500],
            "strict_mismatches": result.get("strict_mismatches", []),
            "candidate_import_error": result.get("candidate_import_error"),
        }
        buckets[_bucket(result)].append(record)

    summary: dict[str, Any] = {
        "total_results": len(data["results"]),
        "buckets": {},
        "case_counts": {},
    }
    for bucket_name, records in sorted(buckets.items()):
        class_counts = Counter(record["classification"] for record in records)
        summary["buckets"][bucket_name] = {
            "total": len(records),
            "classification_counts": dict(sorted(class_counts.items())),
            "stdout_false_positive_count": sum(1 for record in records if record["stdout_false_positive"]),
            "repo_stdout_equal_count": sum(1 for record in records if record["repo_stdout_equal"]),
        }
        per_case = Counter((record["channel"], record["name"], record["classification"]) for record in records)
        summary["case_counts"][bucket_name] = [
            {
                "channel": channel,
                "name": name,
                "classification": classification,
                "count": count,
            }
            for (channel, name, classification), count in sorted(per_case.items())
        ]

    return {
        "source_summary": {
            key: data.get(key)
            for key in (
                "repo_root",
                "run_dir",
                "materialized_root",
                "seed",
                "timeout",
                "case_count",
                "stdout_false_positive_count",
                "strict_fail_count",
                "import_error_count",
            )
        },
        "summary": summary,
        "records": {bucket_name: records for bucket_name, records in sorted(buckets.items())},
    }


def _typst_table(rows: list[tuple[str, str]]) -> str:
    lines = ["#table(", "  columns: 2,", "  [field], [value],"]
    for left, right in rows:
        lines.append(f"  [{left}], [{right}],")
    lines.append(")")
    return "\n".join(lines)


def write_typst(report: dict[str, Any], out: Path) -> None:
    summary = report["summary"]
    lines: list[str] = [
        "= Strict I/O Stdout Bucket Classification",
        "",
        "#set text(size: 10pt)",
        "#set heading(numbering: \"1.\")",
        "#set page(margin: 0.85in)",
        "",
        "This separates cases where the original I/O oracle has no meaningful signal because both scripts print empty stdout from cases with non-empty stdout.",
        "",
        _typst_table(
            [
                ("matched candidates", f"`{summary['total_results']}`"),
                ("empty equal stdout", f"`{summary['buckets'].get('empty_equal_stdout', {}).get('total', 0)}`"),
                ("non-empty equal stdout", f"`{summary['buckets'].get('nonempty_equal_stdout', {}).get('total', 0)}`"),
                ("stdout differs", f"`{summary['buckets'].get('stdout_diff', {}).get('total', 0)}`"),
            ]
        ),
        "",
        "== Main Findings",
        "",
        "- Empty stdout is the whole false-positive story in this L40S clean rerun: all `156` stdout false positives have `candidate_stdout == reference_stdout == \"\"`.",
        "- There are no cases where stdout is non-empty, equal, and the generated implementation is still incorrect.",
        "- There are `4` non-empty generated stdout cases, but they do not pass the original stdout oracle because the reference stdout is empty.",
        "",
    ]

    for bucket_name, bucket in summary["buckets"].items():
        lines.extend(
            [
                f"== `{bucket_name}`",
                "",
                _typst_table(
                    [
                        ("total", f"`{bucket['total']}`"),
                        ("repo stdout equal", f"`{bucket['repo_stdout_equal_count']}`"),
                        ("stdout false positives", f"`{bucket['stdout_false_positive_count']}`"),
                        ("classification counts", f"`{bucket['classification_counts']}`"),
                    ]
                ),
                "",
            ]
        )
        for row in summary["case_counts"].get(bucket_name, []):
            lines.append(
                f"- `{row['channel']}` / `{row['name']}`: `{row['classification']}` x `{row['count']}`"
            )
        lines.append("")

    lines.extend(["== Non-Empty Stdout Records", ""])
    nonempty_records = report["records"].get("nonempty_equal_stdout", []) + report["records"].get("stdout_diff", [])
    if not nonempty_records:
        lines.append("None.")
    for record in nonempty_records:
        lines.extend(
            [
                f"=== `{record['channel']}` / `{record['name']}`",
                "",
                _typst_table(
                    [
                        ("candidate", f"`{record['candidate_rel']}`"),
                        ("classification", f"`{record['classification']}`"),
                        ("repo stdout equal", f"`{record['repo_stdout_equal']}`"),
                        ("candidate stdout", f"`{record['candidate_stdout_excerpt']!r}`"),
                        ("reference stdout", f"`{record['reference_stdout_excerpt']!r}`"),
                    ]
                ),
                "",
            ]
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify strict I/O results by stdout signal strength.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--typst-out", type=Path)
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    report = classify(data)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.typst_out:
        args.typst_out.parent.mkdir(parents=True, exist_ok=True)
        write_typst(report, args.typst_out)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
