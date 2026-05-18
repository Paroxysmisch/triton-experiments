# IR To SMT-LIB Fragment Plan

This note defines the concrete fragment for translating PyTorch and Triton IRs
to SMT-LIB/Z3. The guiding rule is: accept a tiny fragment soundly and reject
everything else.

## Target

Given a PyTorch specification and a generated Triton implementation, emit SMT
queries that prove or refute:

```text
For all valid inputs in the supported fragment,
Triton output equals PyTorch output.
```

The first backend target is standalone SMT-LIB consumed by Z3.

## Overall Pipeline

```text
PyTorch spec
  -> torch.export / FX / ATen graph, eventually
  -> Spec IR

Triton implementation
  -> Triton TTIR / Triton MLIR, eventually
  -> Kernel IR

Spec IR + Kernel IR
  -> SMT-LIB verification conditions
  -> Z3
```

For the local prototype, we may temporarily parse task metadata/source and
generated Python ASTs, but the intended trusted path is normalized IRs.

## Initial PyTorch Spec Fragment

Start with 1D contiguous elementwise tensor functions.

Supported operators:

```text
torch.add(input, other, alpha=alpha)
torch.sub(input, other, alpha=alpha)
torch.mul(input, other)
torch.div(input, other, rounding_mode=None|'floor'|'trunc')
torch.sqrt(input)
torch.relu(input)
torch.tanh(input)
torch.reciprocal(input)
torch.where(cond, a, b)
```

Initial assumptions:

```text
single output tensor
same-shape tensors
contiguous layout
no aliasing
no autograd
no complex values
real arithmetic abstraction for floating point
no dtype promotion at first
```

Next additions, in order:

```text
scalar `other`
alpha parameters
rounding_mode
broadcasting
strides
out=
SMT floating-point sorts for fp32/fp64
```

## Spec IR

Minimal expression language:

```text
Expr ::=
  Input(name, index)
  Scalar(name)
  Const(value)
  Add(Expr, Expr)
  Sub(Expr, Expr)
  Mul(Expr, Expr)
  Div(Expr, Expr)
  FloorDivResult(Expr, Expr)
  TruncDivResult(Expr, Expr)
  Sqrt(Expr)
  Relu(Expr)
  Tanh(Expr)
  Reciprocal(Expr)
  Where(BoolExpr, Expr, Expr)

BoolExpr ::=
  Eq(Expr, Expr)
  Lt(Expr, Expr)
  Le(Expr, Expr)
  And(BoolExpr, BoolExpr)
  Or(BoolExpr, BoolExpr)
  Not(BoolExpr)
```

For `torch.add`:

```text
SpecExpr(i) = Add(Input("input", i), Mul(Scalar("alpha"), Input("other", i)))
```

For `torch.div(..., rounding_mode="floor")`:

```text
SpecExpr(i) = Floor(Div(Input("input", i), Input("other", i)))
```

## Initial Triton Kernel Fragment

Accept only canonical 1D elementwise kernels:

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
simple arithmetic: +, -, *, /, unary -
simple comparisons: <, <=, ==, !=
scalar constexprs
```

Rejected initially:

```text
atomics
reductions
tl.dot / matmul
tl.make_block_ptr
indirect indexing
data-dependent loops
random/dropout
pointer casts
unsupported dtype conversions
possible aliasing
multiple output tensors
non-affine pointer expressions
```

## Kernel IR

The frontend should summarize accepted kernels into:

```json
{
  "fragment": "elementwise_1d_v0",
  "index": "i = pid * BLOCK_SIZE + lane",
  "domain": "0 <= i < n_elements",
  "grid": "ceil(n_elements / BLOCK_SIZE)",
  "loads": [
    {"name": "x", "tensor": "input", "index": "i", "mask": "i < n_elements"},
    {"name": "y", "tensor": "other", "index": "i", "mask": "i < n_elements"}
  ],
  "stores": [
    {"tensor": "out", "index": "i", "value": "x + y", "mask": "i < n_elements"}
  ]
}
```

This summary is the input to SMT generation.

## Verification Obligations

### 1. Memory Safety

Every load/store index is in range when its mask is true.

```text
forall pid lane.
  active(pid, lane) -> 0 <= index(pid, lane) < N
```

### 2. Coverage

Every output index in the logical output domain is written.

```text
forall i.
  0 <= i < N ->
  exists pid lane.
    0 <= pid < grid_size
    and 0 <= lane < BLOCK_SIZE
    and i = pid * BLOCK_SIZE + lane
```

For the canonical grid:

```text
grid_size = ceil(N / BLOCK_SIZE)
```

### 3. No Duplicate Conflicting Writes

No two active program instances write different values to the same output
location.

```text
forall pid1 lane1 pid2 lane2.
  active(pid1,lane1) and active(pid2,lane2)
  and index(pid1,lane1) = index(pid2,lane2)
  ->
  value(pid1,lane1) = value(pid2,lane2)
```

For canonical elementwise kernels this follows from injectivity of
`pid * BLOCK_SIZE + lane` when `0 <= lane < BLOCK_SIZE`.

### 4. Value Equivalence

For every output index, implementation expression equals spec expression.

```text
forall i.
  0 <= i < N -> ImplExpr(i) = SpecExpr(i)
```

## SMT-LIB Encoding

### Sorts

First version:

```text
Int  for shapes/indices
Real for numeric tensor values
Array Int Real for tensors
```

Later:

```text
(_ FloatingPoint 8 24) for fp32
(_ FloatingPoint 11 53) for fp64
BitVec for low-level pointer/index overflow checks
```

### Example: `add` Spec Vs Naive Triton

Declarations:

```smt2
(set-logic AUFLIRA)

(declare-const N Int)
(declare-const BLOCK Int)
(declare-const alpha Real)
(declare-const X (Array Int Real))
(declare-const Y (Array Int Real))

(assert (> N 0))
(assert (> BLOCK 0))
```

Spec:

```smt2
(define-fun spec ((i Int)) Real
  (+ (select X i) (* alpha (select Y i))))
```

Naive implementation:

```smt2
(define-fun impl ((i Int)) Real
  (+ (select X i) (select Y i)))
```

Counterexample query:

```smt2
(assert
  (exists ((i Int))
    (and (<= 0 i)
         (< i N)
         (not (= (impl i) (spec i)))))))

(check-sat)
(get-model)
```

Expected model shape:

```text
N = 1
alpha = 2
X[0] = 0
Y[0] = 1
impl(0) = 1
spec(0) = 2
```

### Proof Query Form

To prove an obligation, assert its negation:

```smt2
(assert (not <obligation>))
(check-sat)
```

Interpretation:

```text
unsat -> obligation proved
sat   -> model is counterexample
unknown -> report inconclusive
```

## Handling Unsupported Features

Every report should include:

```json
{
  "accepted": false,
  "unsupported_features": [
    "tl.dot",
    "tl.make_block_ptr",
    "non-affine pointer expression"
  ]
}
```

No unsupported construct should be approximated as if it were supported.

## Implementation Milestones

### Phase 1: Hand-Fed SMT For `add`

Build:

```text
tools/semantic_checks/smt/ir.py
tools/semantic_checks/smt/emit_smt.py
tools/semantic_checks/smt/check_add.py
```

Goal:

```bash
python3 tools/semantic_checks/smt/check_add.py --emit /tmp/add_equiv.smt2
```

Expected:

```text
Z3 returns counterexample for alpha != 1
```

### Phase 2: Spec Frontend For Simple T Tasks

Lower selected TritonBench-T tasks:

```text
add.py
sub.py
mul.py
div.py
sqrt.py
relu.py
tanh.py
reciprocal.py
```

Initially use metadata/source parsing. Later replace with `torch.export`/ATen.

### Phase 3: Kernel Frontend For Canonical Elementwise Pattern

Parse generated Triton source or TTIR and produce Kernel IR summaries.

First accepted pattern:

```text
linear offsets
masked loads
masked store
one output
same-shape tensors
simple arithmetic expression
```

### Phase 4: Full Obligation Emitter

Emit separate SMT queries for:

```text
memory_safety.smt2
coverage.smt2
no_conflicting_writes.smt2
value_equivalence.smt2
```

### Phase 5: Batch Mining

Run over `LLM_generated/` and produce:

```json
{
  "task": "add.py",
  "jsonl": "...",
  "line": 16,
  "fragment": "elementwise_1d_v0",
  "status": "counterexample",
  "model": {
    "N": 1,
    "alpha": 2,
    "X[0]": 0,
    "Y[0]": 1
  }
}
```

## Soundness Rules

1. Only accepted fragments get proof/counterexample claims.
2. Every assumption is emitted into the SMT file and report.
3. Unsupported operations are rejected.
4. Floating-point mode is explicit:

```text
real_arithmetic
smt_fp32
smt_fp64
```

5. The SMT emitter should have golden output tests for each supported operator.
6. Counterexamples should be reified back into task-level terms whenever
   possible.

