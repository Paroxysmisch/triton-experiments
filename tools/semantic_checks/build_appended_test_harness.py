#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "#" * 146


def gold_test_tail(reference: Path) -> str:
    text = reference.read_text(encoding="utf-8", errors="ignore")
    if MARKER in text:
        return text.rsplit(MARKER, 1)[1].strip() + "\n"
    test_start = text.find("def test_")
    if test_start >= 0:
        return text[test_start:].strip() + "\n"
    raise SystemExit(f"could not find gold test tail in {reference}")


def build(candidate: Path, reference: Path, out: Path) -> None:
    candidate_text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
    tail = gold_test_tail(reference)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(candidate_text + "\n\n" + MARKER + "\n\n" + tail, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a generated implementation + gold test-tail script, matching TritonBench call-accuracy style."
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    build(args.candidate, args.reference, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
