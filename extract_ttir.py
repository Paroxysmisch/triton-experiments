import importlib.util
import sys
import os
import torch
import triton
import argparse
import glob
import time
from unittest.mock import patch


def get_ttir(file_path, function_name, construct_script):
    # 1. Identify Inductor's Cache Dir
    # Inductor uses /tmp/torchinductor_<user> by default
    import getpass

    user = getpass.getuser()
    base_cache = f"/tmp/torchinductor_{user}"

    # Enable the dump and force recompile
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    # Record the current time to find the newest file
    start_time = time.time()

    try:
        DEVICE = triton.runtime.driver.active.get_active_torch_device()
    except Exception:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Import user function
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(file_path))
    spec.loader.exec_module(module)
    target_fn = getattr(module, function_name)

    # 3. Prepare inputs
    env = {"torch": torch, "triton": triton, "DEVICE": DEVICE}
    exec(construct_script, env)
    inputs = env.get("args")
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)

    # 4. Trigger Compilation
    print(f"--- Compiling {function_name} ---", file=sys.stderr)
    compiled_fn = torch.compile(target_fn)
    try:
        compiled_fn(*inputs)
    except Exception as e:
        print(f"Notice: {e}", file=sys.stderr)

    # 5. Search for the TTIR in Inductor's specific triton subfolder
    search_path = os.path.join(base_cache, "triton/**/*.ttir")
    ttir_files = glob.glob(search_path, recursive=True)

    if not ttir_files:
        print(f"Error: No .ttir files found in {base_cache}/triton", file=sys.stderr)
        return

    # 6. Find the most recently modified .ttir file
    # (The one we just generated)
    newest_file = max(ttir_files, key=os.path.getmtime)

    # Just to be safe, check it was created after we started the script
    if os.path.getmtime(newest_file) < (start_time - 1):
        print(
            "Error: The found .ttir file is old. Compilation might have failed.",
            file=sys.stderr,
        )
        return

    # 7. Final Output
    print(f"--- Found TTIR: {newest_file} ---", file=sys.stderr)
    with open(newest_file, "r") as f:
        # Send only the IR to STDOUT
        print(f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("func")
    parser.add_argument("-c", "--construct", required=True)
    args = parser.parse_args()
    get_ttir(args.file, args.func, args.construct)
