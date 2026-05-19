= TritonBench-G L40S Suspect False Positives

#set text(size: 10pt)
#set heading(numbering: "1.")
#set page(margin: 1in)

This note analyzes two TritonBench-G cases from the L40S artifact run:
`fused_recurrent_retention` and `l2_norm_bwd`.

The run inspected here is:

- Repository: `Paroxysmisch/triton-experiments`
- Branch: `mantas-tritonbench`
- Artifact commit: `b75796e4`
- Run directory: `experiments/tritonbench_l40s_20260518_184231`

Important provenance caveat: there are multiple generated implementations for
each benchmark task because TritonBench ran the same task through many model and
prompt variants. In the copied artifact, those variants appear as separate
directories under:

```text
materialized_candidates/<model-or-prompt-family>/candidate_<index>.py
```

For these two tasks, the candidate indices are stable:

#table(
  columns: 3,
  [task], [candidate index], [why],
  [`fused_recurrent_retention.py`], [`candidate_0095.py`], [row 95 in each G JSONL maps to this source file],
  [`l2_norm_bwd.py`], [`candidate_0175.py`], [row 175 in each G JSONL maps to this source file],
)

Thus `candidate_0095.py` is not one implementation. It is a family of
implementations: one from DeepSeek, one from Qwen, one from GPT-4o, one from
Claude, and so on.

The original G execution-accuracy script does the following for a single model
folder:

```python
result1 = subprocess.run([python, generated_file], capture_output=True, text=True)
result2 = subprocess.run([python, gold_file], capture_output=True, text=True)
return result1.stdout == result2.stdout
```

So the implementation tested by I/O is whichever generated file is present in
the model folder being evaluated, paired with the gold file of the same name.
The L40 artifact summary records aggregate pass/fail counts and failure logs,
but it does not retain a per-model, per-task pass matrix for successful files.
The performance JSONs likewise contain task-level timings and do not identify
which model directory supplied the timed candidate. Therefore, the model tables
below show the available generated implementations for the task index, while
the reported speedup is task-level.

What we can still show precisely is:

- the gold implementation and exact gold-side tests that were appended;
- the generated implementations present for the same task index;
- why stdout I/O can mark them as passing;
- deterministic counterexample tests that would fail for concrete generated
  implementations.

== Why TritonBench-G Is Interesting

TritonBench-G is generally more realistic than TritonBench-T. The G channel is
closer to real Triton kernels and GPU-operator implementations, whereas many T
tasks are direct wrappers around PyTorch APIs. However, G still uses synthetic
benchmark inputs and a weak execution-accuracy oracle: many files build a result
dictionary but do not print it or assert it. If both reference and candidate
produce empty stdout, the repo's stdout-only I/O check can classify the candidate
as correct even when returned tensors or side effects differ.

For the two cases below, the source tests assign `result_gold = ...` but do not
print the result. A generated implementation can compute wrong tensors and still
emit the same empty stdout as the gold script.

== Case 1: `fused_recurrent_retention`

=== Source And Benchmark

Reference source:

```text
data/TritonBench_G_v1/fused_recurrent_retention.py
```

Materialized candidate index:

```text
candidate_0095.py
```

Reported L40S speedup:

```text
1.6245x
```

Generated performance timings:

#table(
  columns: 2,
  [T], [generated ms],
  [`4`], [`0.011264`],
  [`8`], [`0.014336`],
  [`16`], [`0.020416`],
  [`32`], [`0.031872`],
  [`64`], [`0.056320`],
  [`128`], [`0.102400`],
  [`256`], [`0.199680`],
)

Golden timings for the same first seven sizes start at `0.017920 ms` and reach
`0.300320 ms` at `T=256`, which explains the reported speedup.

=== Tests It Was Checked On

The source test function is `test_fused_recurrent_retention_with_backward()`.
It creates:

```python
batch_size = 2
n_heads = 4
seq_len = 8
d_head_qk = 16
d_head_v = 16
q, k, v = random CUDA float32 tensors with requires_grad=True
```

It checks four behavioral modes, but only stores the results in a dictionary:

```python
result_gold = test_fused_recurrent_retention_with_backward()
```

The four modes are:

#table(
  columns: 4,
  [case], [initial_state], [output_final_state], [observed fields],
  [`1`], [`None`], [`False`], [`output_shape`, `final_state`, `loss`, gradient norms],
  [`2`], [`present`], [`False`], [`output_shape`, `final_state`, `loss`, gradient norms],
  [`3`], [`present`], [`True`], [`output_shape`, `final_state_shape`, `loss`, gradient norms],
  [`4`], [`None`], [`True`], [`output_shape`, `final_state_shape`, `loss`, gradient norms],
)

None of these stored fields are printed or asserted by the file. Under a
stdout-only harness, both reference and candidate can print nothing and still be
classified as passing.

The artifact has no success log for this file because success logs were not
retained. The available data only shows that `fused_recurrent_retention.py` is
not in `io_check/summary.json`'s failure list. So we know the aggregate I/O run
classified this file as successful, but we do not have a saved stdout/stderr
record saying "model X, candidate_0095.py, test cases 1--4 passed".

The tests themselves are still recoverable because the evaluation script appends
the gold file's test tail to each generated implementation. In other words, each
generated `candidate_0095.py` is tested by running the same four calls above
after the generated code body.

=== What The Correct Kernel Computes

The reference recurrence uses a head-dependent decay:

```python
b_b = 1 - exp2(-5 - head)
h_t = b_b * h_{t-1} + outer(k_t, v_t)
o_t = sum_k h_t[k, v] * q_t[k] * scale
```

For `head = 0`, the reference decay is:

```text
1 - 2^-5 = 0.96875
```

The wrapper also accepts:

```python
fused_recurrent_retention(q, k, v, initial_state=None, output_final_state=False)
```

and must support `output_final_state=True` even when `initial_state is None`.

=== Gold Versus A Concrete Synthesized Implementation

The side-by-side below uses the strongest concrete suspect found in the artifact:

```text
materialized_candidates/Bench_G_general_purpose/output_DeepSeek-R1_comp/candidate_0095.py
```

This does not prove that DeepSeek-R1 was the implementation timed for the
reported performance number. It does prove that at least one generated
implementation for the successful task index is semantically incompatible with
the gold contract.

#table(
  columns: 2,
  [gold source], [synthesized implementation],
  [
```python
# data/TritonBench_G_v1/fused_recurrent_retention.py
b_b = 1 - exp2(-5 - head)
h = b_b * h + outer(k_t, v_t)

def fused_recurrent_retention(
    q, k, v,
    initial_state=None,
    output_final_state=False,
):
    ...
```
  ],
  [
```python
# output_DeepSeek-R1_comp/candidate_0095.py
gamma = 1.0 / (2 ** (head + 1))
h = h * gamma + outer

def fused_recurrent_retention(
    q, k, v,
    initial_state=None,
):
    ...
```
  ],
)

For `head = 0`, the gold decay is `0.96875`; the synthesized decay is `0.5`.
That is not a tolerance issue. It changes the recurrence.

The I/O tests can still pass because the test cases exercise calls and backward
passes but do not compare the returned tensors against an external oracle. In a
stdout-only classifier, this is the effective observation:

#table(
  columns: 4,
  [test], [gold action], [candidate action], [stdout observed],
  [`1`], [compute output and gradients with no final state], [may compute different output], [empty on both sides],
  [`2`], [compute with initial state], [may compute different output], [empty on both sides],
  [`3`], [request final state with initial state], [candidate may reject or mishandle depending on model], [not retained for successes],
  [`4`], [request final state with no initial state], [candidate above lacks the keyword], [not retained for successes],
)

If a concrete candidate rejects tests 3 or 4, it should fail call accuracy. If
it was nevertheless retained as a "success", that means a different model
variant likely supplied the passing implementation for the aggregate run, or the
artifact's success/failure summary is not linked tightly enough to identify the
variant. This is exactly why the next audit needs a per-candidate structured
rerun rather than task-level summaries.

=== Candidate Family

#table(
  columns: 3,
  [model directory], [public signature / generated shape], [diagnosis],
  [`Bench_G_coder_specofic/deepseek_094`], [`fused_recurrent_retention(q, k, v, o, h=None, scale=1.0, i_h=0, ...)`], [syntax error; not source-compatible; no `output_final_state`],
  [`Bench_G_coder_specofic/deepseek_rag_145`], [no public function], [syntax error / no-op fragments],
  [`Bench_G_coder_specofic/deepseek_rag_303`], [no public function], [syntax error / incomplete],
  [`Bench_G_coder_specofic/deepseek_tune_255`], [no public function], [not source-compatible],
  [`Bench_G_coder_specofic/qwen_092`], [`fused_recurrent_retention(q, k, v, o, h, do, i_h, ...)`], [no-op fragments; not source-compatible],
  [`Bench_G_coder_specofic/qwen_tune_300`], [candidate has final-state text], [needs runtime check],
  [`Bench_G_coder_specofic/qwen_tune_rag_250`], [`fused_recurrent_retention(q, k, v, initial_state=None, output_final_state=False, ...)`], [syntax error in materialized file],
  [`Bench_G_general_purpose/Qwen2.5-72B-Instruct_comp`], [`fused_recurrent_retention(q, k, v, initial_state=None, store_final_state=False)`], [wrong public API name for final-state flag],
  [`Bench_G_general_purpose/output_DeepSeek-R1_comp`], [`fused_recurrent_retention(q, k, v, initial_state=None)`], [wrong decay and missing `output_final_state`],
  [`Bench_G_general_purpose/output_claude-3-5-sonnet-20240620_comp`], [`fused_recurrent_retention(q, k, v, initial_state=None, store_final_state=False)`], [contains no-op markers; wrong public API],
  [`Bench_G_general_purpose/output_gpt-4o_comp`], [`fused_recurrent_retention(q, k, v, initial_state=None, store_final_state=False)`], [wrong public API name for final-state flag],
  [`Bench_G_general_purpose/output_o1-2024-12-17_comp`], [`fused_recurrent_retention(q, k, v, initial_state=None, scale=1.0, store_final_state=False)`], [syntax error / no-op markers; wrong public API],
)

Many other materialized `candidate_0095.py` files have no public function or
the same missing-flag pattern. The strongest concrete suspect is:

```text
materialized_candidates/Bench_G_general_purpose/output_DeepSeek-R1_comp/candidate_0095.py
```

It uses:

```python
gamma = 1.0 / (2 ** (head + 1))
```

instead of:

```python
gamma = 1 - exp2(-5 - head)
```

For `head = 0`, that changes the recurrence decay from `0.96875` to `0.5`.
This is an obvious semantic mismatch.

=== Why Existing Tests Can Pass

The source test does not assert tensor equality against an independent oracle.
It records shapes, scalar loss values, and gradient norms inside `result_gold`.
If the generated candidate and reference both produce empty stdout, stdout
comparison alone cannot see:

- wrong recurrence decay;
- missing `output_final_state` support;
- `final_state=None` where a tensor is required;
- wrong backward gradients.

Also, the performance artifact does not prove which model candidate was timed.
Therefore, the rigorous conclusion is:

```text
The task-level L40S success is suspicious, and several candidate_0095.py files
are semantically incompatible with the reference. The exact timed candidate must
be checked by structured execution.
```

=== Obvious Failing Test Cases

These tests should fail for the `output_DeepSeek-R1_comp/candidate_0095.py`
candidate and any candidate with the same API/decay issues.

==== Test A: Missing `output_final_state`

```python
import importlib.util
import torch

def load(path):
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ref = load("data/TritonBench_G_v1/fused_recurrent_retention.py")
cand = load("experiments/tritonbench_l40s_20260518_184231/materialized_candidates/Bench_G_general_purpose/output_DeepSeek-R1_comp/candidate_0095.py")

torch.manual_seed(0)
q = torch.randn(1, 1, 3, 4, device="cuda")
k = torch.randn(1, 1, 3, 4, device="cuda")
v = torch.randn(1, 1, 3, 4, device="cuda")

o_ref, fs_ref = ref.fused_recurrent_retention(
    q, k, v, initial_state=None, output_final_state=True
)

# Expected to fail because the candidate signature lacks output_final_state.
o_cand, fs_cand = cand.fused_recurrent_retention(
    q, k, v, initial_state=None, output_final_state=True
)
```

==== Test B: Wrong Decay

If adapting a candidate to accept the flag, compare outputs:

```python
torch.testing.assert_close(o_cand, o_ref, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(fs_cand, fs_ref, rtol=1e-3, atol=1e-3)
```

The mismatch should be large because the first head uses `0.5` rather than
`0.96875` as the decay.

== Case 2: `l2_norm_bwd`

=== Source And Benchmark

Reference source:

```text
data/TritonBench_G_v1/l2_norm_bwd.py
```

Materialized candidate index:

```text
candidate_0175.py
```

Reported L40S speedup:

```text
1.4409x
```

Generated performance timings:

#table(
  columns: 2,
  [N], [generated ms],
  [`16`], [`0.004992`],
  [`32`], [`0.004992`],
  [`64`], [`0.005120`],
  [`128`], [`0.005120`],
  [`256`], [`0.005376`],
  [`512`], [`0.005120`],
  [`1024`], [`0.006144`],
  [`2048`], [`0.006144`],
  [`4096`], [`0.006144`],
  [`8192`], [`0.007168`],
  [`16384`], [`0.009216`],
)

Golden timings over the same sizes start around `0.006624 ms` and reach
`0.013120 ms` at `N=16384`.

=== Tests It Was Checked On

The source test creates random CUDA tensors and stores returned gradients:

```python
result_gold = test_l2_norm_bwd()
```

The cases are:

#table(
  columns: 3,
  [case], [x shape], [dy shape],
  [`1`], [`(4, 8)`], [`(4, 8)`],
  [`2`], [`(2, 16)`], [`(2, 16)`],
  [`3`], [`(8, 8)`], [`(8, 8)`],
  [`4`], [`(1, 8)`], [`(1, 8)`],
)

Again, the tensor results are not printed or asserted.

The artifact has no retained success log for this file. The evidence of I/O
success is that `l2_norm_bwd.py` is absent from the G failure list in
`io_check/summary.json`. The exact test inputs are recoverable from the gold
file, because G evaluation appends this gold test tail to every generated
`candidate_0175.py`, but the artifact does not say which model variants passed
those tests.

=== What The Correct Kernel Computes

The reference wrapper is:

```python
def _l2_norm_bwd(x, dy, eps=1e-5):
```

It reshapes over the last dimension and computes:

```python
var = sum(x * x)
rstd = 1 / sqrt(var + eps)
dx = dy * rstd - sum(dy * x) * (1 / (var + eps)) * rstd * x
```

This is the backward formula for L2 normalization. The argument order matters:
`x` is the primal input, `dy` is the upstream gradient.

=== Gold Versus Concrete Synthesized Implementations

There are multiple generated `candidate_0175.py` files. Two representative
wrongness modes are visible in the artifact.

#table(
  columns: 2,
  [gold source], [synthesized implementation: DeepSeek-R1 comp],
  [
```python
# data/TritonBench_G_v1/l2_norm_bwd.py
def _l2_norm_bwd(x, dy, eps=1e-5):
    var = sum(x * x)
    rstd = 1 / sqrt(var + eps)
    dx = dy * rstd
       - sum(dy * x) * (1 / (var + eps)) * rstd * x
```
  ],
  [
```python
# output_DeepSeek-R1_comp/candidate_0175.py
def _l2_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    ...
```
  ],
)

The formula inside this DeepSeek-R1 candidate is close to the gold formula, but
the public API order is reversed. The gold tests call `_l2_norm_bwd(x, dy)`.
Under this candidate signature, that binds:

```text
candidate dy = gold x
candidate x  = gold dy
```

For random tensors this is almost never equivalent.

A second wrongness mode appears in Qwen/Claude-style candidates:

#table(
  columns: 2,
  [gold source], [synthesized implementation],
  [
```python
def _l2_norm_bwd(x, dy, eps=1e-5):
    ...
```
  ],
  [
```python
def _l2_norm_bwd(x, dy, eps=1e-6):
    ...

# or
def _l2_norm_bwd(x, dy, eps=1e-12):
    ...
```
  ],
)

Those versions can look correct on ordinary random-normal inputs, because
`sum(x*x)` is usually much larger than either epsilon. They become visibly wrong
for near-zero rows, where the default epsilon controls the scale.

The stdout classifier cannot distinguish these cases. The gold test stores the
four returned tensors in `results`, but the script prints nothing:

#table(
  columns: 4,
  [test], [shape], [what it stores], [stdout observed],
  [`1`], [`(4, 8)`], [`dx` tensor], [empty],
  [`2`], [`(2, 16)`], [`dx` tensor], [empty],
  [`3`], [`(8, 8)`], [`dx` tensor], [empty],
  [`4`], [`(1, 8)`], [`dx` tensor], [empty],
)

=== Candidate Family

#table(
  columns: 3,
  [model directory], [public signature / generated shape], [diagnosis],
  [`Bench_G_coder_specofic/deepseek_094`], [`_l2_norm_bwd(x, dy, eps=1e-12, ...)`], [epsilon differs from source default],
  [`Bench_G_coder_specofic/deepseek_rag_145`], [`_l2_norm_bwd(DX, DY, X, eps=1e-5, ...)`], [wrong API shape],
  [`Bench_G_coder_specofic/deepseek_rag_303`], [`_l2_norm_bwd(x, dy, eps)`], [needs runtime check],
  [`Bench_G_coder_specofic/deepseek_tune_255`], [`_l2_norm_bwd(x, dy, eps=1e-5, out=None)`], [plausible on main path],
  [`Bench_G_coder_specofic/qwen_092`], [no `_l2_norm_bwd`], [syntax/no-op fragments],
  [`Bench_G_general_purpose/Qwen2.5-72B-Instruct_comp`], [`_l2_norm_bwd(x, dy, eps=1e-6)`], [epsilon differs],
  [`Bench_G_general_purpose/Qwen2.5-72B-Instruct_simp`], [`_l2_norm_bwd(x, dy, eps=1e-6)`], [2D-only; epsilon differs],
  [`Bench_G_general_purpose/output_DeepSeek-R1_comp`], [`_l2_norm_bwd(dy, x, eps)`], [argument order is reversed],
  [`Bench_G_general_purpose/output_claude-3-5-sonnet-20240620_simp`], [`_l2_norm_bwd(x, dy, eps=1e-12)`], [2D-only; epsilon differs],
  [`Bench_G_general_purpose/output_gpt-4o_comp`], [`_l2_norm_bwd(x, dy, eps=1e-5)`], [contains no-op markers but public formula looks plausible],
  [`Bench_G_general_purpose/output_o1-2024-12-17_comp`], [`_l2_norm_bwd(x, dy, eps=1e-5)`], [2D-only],
)

The strongest concrete suspect is:

```text
materialized_candidates/Bench_G_general_purpose/output_DeepSeek-R1_comp/candidate_0175.py
```

It defines:

```python
def _l2_norm_bwd(dy: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
```

The source-style call `_l2_norm_bwd(x, dy)` will bind:

```text
candidate dy = source x
candidate x  = source dy
```

That computes the wrong gradient.

Another class of candidates uses `eps=1e-12` instead of the source default
`eps=1e-5`. This may pass ordinary random inputs but fails near zero-norm rows.

=== Why Existing Tests Can Pass

The existing source tests store tensors in `result_gold` but do not print them.
Thus a candidate can be wrong while stdout remains empty.

Additionally, the random test inputs are ordinary normal values. They are not
designed to expose epsilon-sensitive cases such as near-zero rows.

=== Obvious Failing Test Cases

==== Test A: Argument Order Reversal

This catches `output_DeepSeek-R1_comp/candidate_0175.py`.

```python
import importlib.util
import torch

def load(path):
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ref = load("data/TritonBench_G_v1/l2_norm_bwd.py")
cand = load("experiments/tritonbench_l40s_20260518_184231/materialized_candidates/Bench_G_general_purpose/output_DeepSeek-R1_comp/candidate_0175.py")

x = torch.tensor([[1.0, 2.0, -1.0, 0.5]], device="cuda")
dy = torch.tensor([[0.25, -0.75, 0.5, 1.0]], device="cuda")

dx_ref = ref._l2_norm_bwd(x, dy, eps=1e-5)
dx_cand = cand._l2_norm_bwd(x, dy, eps=1e-5)

torch.testing.assert_close(dx_cand, dx_ref, rtol=1e-4, atol=1e-4)
```

The assertion should fail for a reversed-argument implementation.

==== Test B: Epsilon Sensitivity

This catches candidates that use `eps=1e-12` when called with defaults.

```python
x = torch.full((2, 8), 1e-8, device="cuda")
dy = torch.arange(16, dtype=torch.float32, device="cuda").reshape(2, 8)

dx_ref = ref._l2_norm_bwd(x, dy)       # eps defaults to 1e-5
dx_cand = cand._l2_norm_bwd(x, dy)     # suspect default may be 1e-12

torch.testing.assert_close(dx_cand, dx_ref, rtol=1e-4, atol=1e-4)
```

The outputs differ substantially because `1 / sqrt(1e-5)` and
`1 / sqrt(1e-12)` are separated by orders of magnitude.

== Summary

#table(
  columns: 5,
  [case], [speedup], [strongest wrongness], [why stdout tests pass], [best counterexample],
  [`fused_recurrent_retention`], [`1.6245x`], [wrong decay and missing `output_final_state` API], [result dictionary is not printed/asserted], [`output_final_state=True`, especially with `initial_state=None`],
  [`l2_norm_bwd`], [`1.4409x`], [some candidates reverse `(x, dy)` or use wrong epsilon], [returned tensors are not printed/asserted], [asymmetric `x, dy` pair; near-zero `x` for epsilon],
)

The most rigorous next step is to run the counterexample snippets on the L40S
environment against every materialized `candidate_0095.py` and `candidate_0175.py`
variant, because the current artifact records task-level speedup but not exact
model-candidate provenance for the timed performance result.
