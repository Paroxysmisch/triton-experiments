#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


RESULT_GLOBALS = ("test_results", "result_gold", "results")


HARNESS = r'''

# ---- TritonBench structured stdout hardening ----
def __tb_to_jsonable(value):
    import math

    try:
        import torch
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()
            return {
                "__tensor__": True,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "values": tensor.tolist(),
            }
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): __tb_to_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [__tb_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    return repr(value)


def __tb_emit_structured_result():
    import json

    for name in ("test_results", "result_gold", "results"):
        if name in globals():
            payload = {"result_name": name, "result": __tb_to_jsonable(globals()[name])}
            print("__TRITONBENCH_RESULT__=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
            return
    print("__TRITONBENCH_RESULT_MISSING__")


__tb_emit_structured_result()
# ---- end TritonBench structured stdout hardening ----
'''


def has_result_global(text: str) -> bool:
    return any(f"{name} =" in text or f"{name}=" in text for name in RESULT_GLOBALS)


def should_harden(text: str, only_if_result_global: bool) -> bool:
    if "__TRITONBENCH_RESULT__" in text:
        return False
    if only_if_result_global and not has_result_global(text):
        return False
    return True


def harden_file(src: Path, dst: Path, only_if_result_global: bool) -> bool:
    text = src.read_text(encoding="utf-8", errors="ignore")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if should_harden(text, only_if_result_global):
        dst.write_text(text.rstrip() + "\n" + HARNESS + "\n", encoding="utf-8")
        return True
    shutil.copy2(src, dst)
    return False


def iter_python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.py"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create hardened benchmark copies that print canonical structured result globals."
    )
    parser.add_argument("--src", required=True, type=Path, help="Input Python file or directory")
    parser.add_argument("--dst", required=True, type=Path, help="Output Python file or directory")
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Append the structured-result emitter to every file, even if no result global is statically visible.",
    )
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    only_if_result_global = not args.all_files

    changed = 0
    total = 0
    if src.is_file():
        changed += harden_file(src, dst, only_if_result_global)
        total = 1
    else:
        for file in iter_python_files(src):
            rel = file.relative_to(src)
            changed += harden_file(file, dst / rel, only_if_result_global)
            total += 1

    print(f"hardened {changed} / {total} files into {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
