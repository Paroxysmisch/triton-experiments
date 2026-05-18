# Codex Resume Prompt: TritonBench Semantic Verification Experiment

You are working in `/Users/baksy/Repositories/TritonBench`.

## Goal

Explore whether Python-specific verification tools, especially Nagini/Viper-style verification, can help verify Triton kernels semantically rather than relying only on randomized runtime correctness tests.

The motivating target is KernelBench-style tasks: a PyTorch reference specification, a generated or hand-written optimized kernel for specific hardware, and a need to prove the optimized implementation satisfies the original spec. TritonBench is the current local substrate because it provides many Triton operators and PyTorch-aligned operator specs.

## Local Repo State

Main repo:

- Path: `/Users/baksy/Repositories/TritonBench`
- Remote: `https://github.com/thunlp/TritonBench`
- Current branch: `main`
- Last observed commit: `603e28a` (`fix leakeage in tests`)
- The main repo has an untracked `hf_datasets/` directory containing Hugging Face dataset clones.

Hugging Face dataset clones:

- `/Users/baksy/Repositories/TritonBench/hf_datasets/tritonbench_g`
  - Remote: `https://huggingface.co/datasets/LiShangZ/tritonbench_g`
  - Branch: `master`
  - Contains `TritonBench_G_v1.json`
  - 184 entries
- `/Users/baksy/Repositories/TritonBench/hf_datasets/tritonbench_t`
  - Remote: `https://huggingface.co/datasets/LiShangZ/tritonbench_t`
  - Branch: `master`
  - Contains `TritonBench_T_v1.json`
  - 166 entries

Important local directories:

- `data/TritonBench_G_v1/`: 184 executable Python files with real-world Triton operators.
- `data/TritonBench_T_v1/`: 166 executable Python files with PyTorch-interface-aligned reference functions/tests.
- `LLM_generated/`: JSONL outputs from models used in the paper.
- `EVAL/eval_G/` and `EVAL/eval_T/`: original evaluation scripts.
- `hf_datasets/tritonbench_g/TritonBench_G_v1.json`: metadata and code for the G channel.
- `hf_datasets/tritonbench_t/TritonBench_T_v1.json`: semantic metadata for the T channel.

## Papers And Sources Already Consulted

- TritonBench paper: https://arxiv.org/abs/2502.14752
- KernelBench paper: https://arxiv.org/abs/2502.10517
- TritonBench GitHub: https://github.com/thunlp/TritonBench
- KernelBench GitHub: https://github.com/ScalingIntelligence/KernelBench
- Nagini overview: https://www.pm.inf.ethz.ch/research/nagini.html
- Hugging Face collection: https://huggingface.co/collections/LiShangZ/tritonbench

Important paper takeaways:

- TritonBench has two channels:
  - TritonBench-G: real-world GitHub Triton operators, 184 entries.
  - TritonBench-T: PyTorch-interface-aligned operator tasks, 166 entries.
- TritonBench evaluates call accuracy, execution accuracy, speedup, and GPU efficiency. G also uses code similarity.
- KernelBench tasks are PyTorch `Model` specs plus fixed input generators; generated code implements `ModelNew`.
- KernelBench correctness is currently randomized testing against PyTorch outputs, commonly five random inputs, plus runtime speed measurement.
- The KernelBench paper explicitly identifies stronger/formal verification as an area for future work.
- Nagini is a verifier for statically typed Python programs based on Viper, but raw `@triton.jit` is Python-shaped DSL code, not ordinary Python. Do not assume Nagini can verify Triton directly.

## Working Hypothesis

Do translation validation, not direct verification of arbitrary Triton.

Suggested architecture:

1. Pick a small TritonBench-T task whose PyTorch spec is simple.
2. Extract a pure, bounded Python model from the PyTorch spec.
3. Parse or pattern-match the Triton kernel into a Nagini-friendly Python loop/memory model.
4. Prove:
   - wrapper/interface matches the spec,
   - launch grid covers exactly the output domain,
   - masks prevent out-of-bounds loads/stores,
   - each output element equals the PyTorch spec for the supported shape/dtype domain.
5. Keep runtime tests as a final backstop, but make them complementary to semantic checks.

Recommended first targets from TritonBench-T:

- `sqrt.py`
- `tanh.py`
- `sub.py`
- `relu_sqrt.py`
- possibly `add.py` from `data/TritonBench_T_v1/add.py` even if not at the top of the HF JSON ordering

Avoid first:

- attention kernels,
- dropout/training-mode specs,
- convolutions,
- SVD/eig/linear algebra solvers,
- indirect memory access,
- mixed precision or approximate special functions.

## Useful Schema Facts

`hf_datasets/tritonbench_g/TritonBench_G_v1.json` fields:

- `file`
- `repo`
- `simp_instru`
- `simp_instru_len`
- `comp_instru`
- `comp_instru_len`
- `output`
- `output_triton_len`
- `star`
- `difficulty`

`hf_datasets/tritonbench_t/TritonBench_T_v1.json` fields:

- `name`
- `func_inputs`
- `description`
- `math`
- `example`
- `torch_code`
- `torch_cnt`
- `other`
- `difficulty`
- `params_cnt`
- `file`

Observed difficulty distribution:

TritonBench-T:

- difficulty 1: 22
- difficulty 2: 37
- difficulty 3: 54
- difficulty 4: 49
- difficulty 5: 4

TritonBench-G:

- difficulty 1: 3
- difficulty 2: 27
- difficulty 3: 65
- difficulty 4: 84
- difficulty 5: 5

## Evaluation Script Caveats

The original eval scripts hard-code the authors' Python interpreter path:

- `EVAL/eval_G/0_call_acc.py`
- `EVAL/eval_G/1_exe_acc.py`
- `EVAL/eval_T/0_call_acc.py`
- `EVAL/eval_T/1_exe_acc.py`

They assume CUDA GPUs and use paths like:

```python
py_interpreter = "/home/lijianling/miniconda3/envs/LLM/bin/python"
```

Patch these before trying local execution. Do not run large GPU evaluation blindly.

## Initial Implementation Plan

Create a small `semantic_checks/` or `tools/semantic_checks/` prototype with:

1. Dataset loader:
   - Read the HF T JSON.
   - Join entries to `data/TritonBench_T_v1/<file>`.
   - Select difficulty 1-2 tasks with simple `torch_code`.

2. Static interface checker:
   - Parse the Python file with `ast`.
   - Confirm target function name and arguments from HF metadata.
   - Identify `@triton.jit` kernels if present.
   - Extract wrapper-to-kernel launch calls.

3. Triton pattern checker for elementwise kernels:
   - Recognize common forms:
     - `pid = tl.program_id(axis=0)`
     - `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
     - `mask = offsets < n_elements`
     - `x = tl.load(ptr + offsets, mask=mask, ...)`
     - `tl.store(out + offsets, expr, mask=mask)`
   - Produce a semantic summary like:
     - domain: `0 <= i < n_elements`
     - reads: `input[i]`, `other[i]`
     - writes: `out[i]`
     - expression: `input[i] + alpha * other[i]`, etc.

4. Nagini-oriented model generation:
   - Generate ordinary typed Python functions over lists/floats/ints.
   - Use preconditions for lengths, bounds, scalar params.
   - Use postconditions for per-element equivalence.
   - Start with integer or real-valued mathematical models; be explicit that IEEE floating point and PyTorch dtype promotion are not fully modeled yet.

5. Report format:
   - `semantic_ok`: boolean
   - `supported_pattern`: boolean
   - `interface_findings`
   - `memory_safety_findings`
   - `equivalence_obligations`
   - `unsupported_features`

## Suggested First Prompt To Continue

Start by implementing the smallest semantic-checking prototype for TritonBench-T difficulty 1 elementwise tasks. Use `uv` if the repo has project config; otherwise use the existing system Python only for local read-only scripts if necessary. Avoid installing heavy GPU dependencies unless asked. Focus on static parsing and report generation first; do not require CUDA.

The first deliverable should be a CLI that can run something like:

```bash
python tools/semantic_checks/check_t_task.py --task sqrt.py
```

or, if a project environment is introduced:

```bash
uv run python tools/semantic_checks/check_t_task.py --task sqrt.py
```

and print a concise JSON or markdown report about interface/spec/kernel-pattern compatibility.

## Notes For Codex

- The user wants semantic checking, not only runtime correctness.
- Be conservative about verification claims. Say "bounded model", "restricted fragment", or "translation validation prototype" unless a real proof has been run.
- Prefer the TritonBench-T dataset first because it has explicit `torch_code`, `math`, and function signatures.
- The main verification challenge is mapping Triton DSL semantics to a verifier-friendly model.
- Do not assume Nagini supports PyTorch/Triton directly.
- Runtime correctness remains useful but should be treated as weaker than semantic/spec checking.
