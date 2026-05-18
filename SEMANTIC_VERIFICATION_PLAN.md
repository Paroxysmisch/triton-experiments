# Semantic Verification Plan for TritonBench

This note summarizes the verification exploration so far and proposes a concrete
path for translating PyTorch and Triton IRs into SMT-LIB/Z3 equivalence checks.

## Goal

We want to check whether a generated Triton implementation is semantically
equivalent to a PyTorch reference specification, not merely whether it passes a
small randomized input-output test set.

The desired theorem is:

```text
For all valid inputs satisfying the task preconditions,
the Triton wrapper/kernel produces the same observable output tensors as the
PyTorch specification.
```

For the initial prototype, "all valid inputs" should be interpreted relative to
a small explicitly supported fragment. Unsupported features should be rejected,
not approximated silently.

## Local Dataset And Artifact Findings

The downloaded Hugging Face dataset clones contain task metadata and reference
code, but not generated correctness classifications.

`hf_datasets/tritonbench_t/TritonBench_T_v1.json` contains 166 tasks with fields:

```text
description, difficulty, example, file, func_inputs, math, name, other,
params_cnt, torch_cnt, torch_code
```

`hf_datasets/tritonbench_g/TritonBench_G_v1.json` contains 184 tasks with fields:

```text
comp_instru, comp_instru_len, difficulty, file, output,
output_triton_len, repo, simp_instru, simp_instru_len, star
```

The local repo also contains raw model generations under `LLM_generated/` and
golden/reference performance artifacts under:

```text
performance_metrics/perf_T/golden_results
performance_metrics/perf_G/golden_results
```

It does not contain generated post-eval artifacts such as:

```text
generated call-accuracy survivor folders
generated IO-accuracy survivor folders
generated performance JSONs
logs showing which generated kernels passed and achieved speedups
```

Those must be produced later on a CUDA-capable Linux machine.

## Evaluation Harness Caveat

TritonBench-T's execution-accuracy script compares only captured stdout:

```text
EVAL/eval_T/1_exe_acc.py
```

Many T reference files build a `test_results` dictionary but do not print tensor
values. Therefore, a generated solution can be semantically wrong yet appear
correct if both generated and reference scripts produce the same empty stdout.

A helper audit script was added:

```text
tools/semantic_checks/audit_io_false_positive.py
```

It extracts one generated JSONL solution, appends the original T test body,
reproduces the stdout-based classifier when runnable, and emits deterministic
counterexamples for known bug families such as ignored `alpha`, missing
broadcasting, ignored `rounding_mode`, and wrong dimension reductions.

Example high-signal counterexample:

```text
Task: torch.add
Generated pattern: out[i] = input[i] + other[i]
Spec:              out[i] = input[i] + alpha * other[i]

Counterexample:
  input = [0.0]
  other = [1.0]
  alpha = 2.0

Expected = [2.0]
Observed = [1.0]
```

## Why Not Alive2 As The Main Tool?

Alive2 is useful for LLVM IR refinement and translation validation, but the
project does not need to verify LLVM optimizations. We are willing to trust the
Triton/LLVM compiler stack for this experiment.

The target theorem is not:

```text
LLVM before optimization refines LLVM after optimization
```

It is:

```text
PyTorch specification equals Triton kernel behavior
```

Alive2 does not directly provide PyTorch or Triton tensor semantics. We would
still need a semantic bridge from PyTorch/Triton to a formal representation.

## Why Not Nagini As The Main Tool?

A PyTorch/Triton-to-vanilla-Python translation followed by Nagini/Viper
verification is plausible for a tiny fragment, especially for index arithmetic,
loop invariants, bounds checks, and simple list models.

However, Nagini does not understand PyTorch tensors or Triton kernels natively.
The hard part remains: producing a sound model of PyTorch and Triton semantics.
Nagini also becomes awkward for floating point, NaN/inf/signed-zero behavior,
dtype promotion, broadcasting, reductions, and block/vector semantics.

Nagini can be useful later as a readable proof artifact or for simple memory
safety/index proofs, but it should not be the primary backend.

## Recommended Direction

Use normalized IRs and translate a strictly whitelisted fragment to SMT-LIB:

```text
PyTorch spec
  -> torch.export / FX / ATen graph
  -> small Spec IR

Triton generated code
  -> Triton TTIR / Triton MLIR
  -> small Kernel IR

Spec IR + Kernel IR
  -> SMT-LIB verification condition
  -> Z3 proof or counterexample
```

The key design principle is aggressive rejection. The verifier should be sound
for the accepted fragment and explicitly report unsupported features elsewhere.

## Why Use PyTorch Export/ATen?

`torch.export` is designed to produce an ahead-of-time graph for tensor
computation. It normalizes Python/PyTorch code into an FX graph containing ATen
operators and records shape constraints. That is a better starting point than
parsing arbitrary Python source.

For early TritonBench-T tasks, we can also use the HF metadata and task files to
identify simple PyTorch specs. But the long-term path should prefer exported
ATen graphs where possible.

## Why Use Triton TTIR/Triton MLIR?

Triton Python source is a Python-shaped DSL, not ordinary Python. The source AST
contains user syntax, decorators, Python control flow, and generated-code
mistakes.

At the Triton IR level, the relevant semantics are more explicit:

```text
program_id
arange
load/store
masks
affine pointer expressions
where
dot/sum/max for later fragments
```

LLVM IR is too low-level for the first equivalence checker because it obscures
the tensor/block structure that is central to Triton kernels.

## Initial Supported Fragment

Start with a tiny but useful fragment:

### PyTorch Spec Fragment

Elementwise contiguous 1D tensor operations:

```text
torch.add
torch.sub
torch.mul
torch.div
torch.sqrt
torch.relu
torch.tanh
torch.reciprocal
torch.where
```

Initial assumptions:

```text
single output tensor
contiguous tensors
same shape
no aliasing
real-number abstraction for floats
no dtype promotion initially
no complex values
no autograd
```

Then add:

```text
scalar other
alpha parameters
rounding_mode for div
broadcasting
strides
out=
```

### Triton Kernel Fragment

Accept kernels of the shape:

```text
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, ...)
y = tl.load(y_ptr + offsets, mask=mask, ...)
out_val = expression(x, y, scalars)
tl.store(out_ptr + offsets, out_val, mask=mask)
```

Supported Triton constructs:

```text
tl.program_id
tl.arange
tl.load
tl.store
tl.where
tl.cdiv
simple arithmetic
simple comparisons
scalar constexprs
```

Reject:

```text
atomics
indirect indexing
data-dependent loops
reductions
dot/matmul
block pointers, initially
random/dropout
pointer casts
unsupported dtype conversions
possible aliasing
```

## Verification Obligations

For accepted kernels, generate SMT obligations for:

### 1. Memory Safety

Every load/store index is in range whenever its mask is true.

```text
forall pid lane.
  active(pid, lane) -> 0 <= offset(pid, lane) < N
```

### 2. Coverage

Every output index is written by some program instance.

```text
forall i.
  0 <= i < N ->
  exists pid lane.
    0 <= lane < BLOCK_SIZE
    and 0 <= pid < grid_size
    and i = pid * BLOCK_SIZE + lane
```

### 3. No Duplicate Conflicting Writes

No two distinct program instances write different values to the same output
index.

```text
forall pid1 lane1 pid2 lane2.
  write_index(pid1,lane1) = write_index(pid2,lane2) ->
  write_value(pid1,lane1) = write_value(pid2,lane2)
```

For the basic 1D elementwise pattern, this can often be strengthened to index
injectivity.

### 4. Value Equivalence

For every logical output element, the implementation expression equals the
PyTorch spec expression.

```text
forall i.
  0 <= i < N ->
  impl_out[i] = spec_out[i]
```

For floating point, start with real arithmetic. Later, add a mode using SMT
floating-point sorts for IEEE fp32/fp64.

## SMT Encoding Sketch

Represent tensors as arrays:

```smt2
(declare-const X (Array Int Real))
(declare-const Y (Array Int Real))
(declare-const OutImpl (Array Int Real))
(declare-const N Int)
(declare-const BLOCK Int)
(declare-const alpha Real)
```

For `torch.add`:

```smt2
(define-fun spec ((i Int)) Real
  (+ (select X i) (* alpha (select Y i))))
```

For a generated naive Triton `x + y`:

```smt2
(define-fun impl ((i Int)) Real
  (+ (select X i) (select Y i)))
```

Equivalence query:

```smt2
(assert (> N 0))
(assert (not
  (forall ((i Int))
    (=> (and (<= 0 i) (< i N))
        (= (impl i) (spec i)))))))
(check-sat)
(get-model)
```

For the ignored-alpha bug, Z3 can produce a model where `alpha != 1`.

## Counterexample-First Workflow

The prototype should produce either:

```json
{
  "status": "proved",
  "fragment": "elementwise_1d_v0",
  "obligations": [...]
}
```

or:

```json
{
  "status": "counterexample",
  "model": {
    "N": 1,
    "alpha": 2,
    "X[0]": 0,
    "Y[0]": 1
  },
  "expected": "2",
  "observed": "1"
}
```

This is stronger than randomized IO testing and gives concrete evidence for
semantic gaps.

## Concrete Implementation Plan

### Phase 0: Preserve The Existing Audit Tool

Keep:

```text
tools/semantic_checks/audit_io_false_positive.py
```

It is useful for mining examples and explaining why stdout IO testing is weak.

### Phase 1: Manual Semantic IR For One Task

Implement a minimal hand-fed checker for `add.py`:

```text
tools/semantic_checks/smt/
  ir.py
  emit_smt.py
  check_add.py
```

Inputs:

```text
spec expression: input + alpha * other
impl expression: input + other
```

Output:

```text
SMT-LIB file
Z3 result
counterexample model
```

This proves the end-to-end shape before touching compiler IRs.

### Phase 2: PyTorch Spec Frontend

Implement a small PyTorch/ATen spec frontend:

```text
torch.add -> SpecExpr.Add(x, Mul(alpha, y))
torch.sub -> SpecExpr.Sub(x, Mul(alpha, y))
torch.div -> Div plus optional rounding mode
relu/sqrt/tanh -> unary expressions
```

For now, use TritonBench-T metadata and simple AST parsing. Later, replace this
with `torch.export`/ATen when working in a CUDA/PyTorch-capable environment.

### Phase 3: Triton Kernel Frontend

Initially parse generated source AST for the canonical elementwise pattern. This
is not the final trusted path, but it lets us iterate locally.

Then replace or augment it with TTIR/Triton MLIR extraction on a CUDA/Linux
machine.

The accepted summary should be explicit:

```json
{
  "domain": "0 <= i < n_elements",
  "reads": ["input[i]", "other[i]"],
  "writes": ["out[i]"],
  "expr": "input[i] + other[i]",
  "grid": "ceil(n_elements / BLOCK_SIZE)"
}
```

### Phase 4: SMT-LIB Emitter

Emit standalone `.smt2` files with:

```text
preconditions
shape/domain assumptions
memory-safety obligations
coverage obligations
value-equivalence obligation
```

Use solver mode:

```text
assert not(obligation)
check-sat
get-model
```

If `unsat`, the obligation is proved.
If `sat`, the model is a counterexample.

### Phase 5: Batch Mining

Run over `LLM_generated/` for supported tasks and report:

```text
task
model output file
line
fragment accepted/rejected
proved/counterexample
counterexample summary
stdout-IO false-positive risk
```

Later, on CUDA Linux, join this with actual generated pass/speedup artifacts.

## Risks And Mitigations

### Risk: Unsound Translation

Mitigation:

```text
tiny fragment
auditable semantic rules
aggressive rejection
golden SMT snapshots
negative tests
explicit assumptions in every report
```

### Risk: Floating-Point Semantics

Mitigation:

Start with real arithmetic and label it clearly:

```text
semantic_mode = "real_arithmetic"
```

Then add:

```text
semantic_mode = "smt_fp32"
```

for selected kernels.

### Risk: PyTorch Surface Is Huge

Mitigation:

Do not try to cover all TritonBench-T. Support a small operator set and produce
useful rejected/unsupported reports.

### Risk: Triton Surface Is Huge

Mitigation:

Accept one canonical elementwise pattern first. Add reductions and matmul only
after the first fragment works well.

## Near-Term Deliverable

The next concrete deliverable should be:

```bash
python3 tools/semantic_checks/smt/check_task.py \
  --jsonl LLM_generated/Bench_T_coder_specific/qwen_full_rag.jsonl \
  --line 16 \
  --task-file add.py \
  --emit /tmp/add_equiv.smt2
```

Expected output:

```text
status: counterexample
N = 1
alpha = 2
input[0] = 0
other[0] = 1
spec[0] = 2
impl[0] = 1
```

That would establish the first end-to-end formal counterexample for a generated
solution that the existing stdout IO harness can miss.
