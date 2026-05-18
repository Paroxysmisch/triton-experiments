# Lean Verification Exploration

This note isolates the Lean option from the SMT-LIB implementation plan. The
main conclusion is that Lean is valuable as a trust/audit layer for a small
semantic core, but it is probably not the fastest primary engine for finding
counterexamples across TritonBench.

## Goal

Explore whether PyTorch/Triton semantic equivalence can be represented and
proved in Lean.

The target theorem shape is:

```text
For all valid inputs satisfying preconditions,
run_triton_kernel(inputs) = run_pytorch_spec(inputs)
```

For Lean, this should be stated over a small formal semantic IR, not over raw
PyTorch or raw Triton source.

## Recommended Lean Boundary

Do not translate arbitrary PyTorch/Triton directly into Lean.

Use:

```text
PyTorch/Triton source or IR
  -> small semantic IR
  -> Lean AST/data definitions
  -> Lean theorem/proof obligation
```

Lean should reason about the semantic IR. The trusted part is then the formal
semantics of that IR plus the theorem statement.

## Minimal Lean Model

Start with tensors as finite functions:

```lean
Tensor α n := Fin n -> α
```

For early arithmetic, use `Int` or `Rat`/`Real`-like abstractions before
attempting IEEE floating point.

Possible core definitions:

```lean
def Tensor (α : Type) (n : Nat) := Fin n -> α

def torchAdd
  (x y : Tensor Int n)
  (alpha : Int) : Tensor Int n :=
  fun i => x i + alpha * y i

def tritonAddNaive
  (x y : Tensor Int n) : Tensor Int n :=
  fun i => x i + y i
```

Then prove or refute:

```lean
theorem triton_add_correct_when_alpha_one :
  ∀ (x y : Tensor Int n),
    torchAdd x y 1 = tritonAddNaive x y

theorem triton_add_not_correct_for_all_alpha :
  ∃ n x y alpha,
    torchAdd x y alpha ≠ tritonAddNaive x y
```

## Triton Execution Model In Lean

For a kernel-like model, represent program instances explicitly:

```text
pid  : Nat
lane : Fin BLOCK
i    = pid * BLOCK + lane
mask = i < N
```

Model masked stores as producing an output tensor:

```text
out[i] = expr(input[i]) if some active pid/lane writes i
```

Core theorems:

```text
coverage:
  every i < N is written by some pid/lane

no_oob:
  every active load/store index is < N

no_duplicate_conflict:
  if two program instances write same i, values agree

value_equivalence:
  written value equals spec value for every i < N
```

## What Lean Is Good For

Lean is strong for:

```text
explicit denotational semantics
auditable theorem statements
index arithmetic
coverage proofs
mask/no-out-of-bounds proofs
finite tensor extensionality
small counterexample theorems
```

Lean is less convenient for:

```text
rapid counterexample search
large batches of generated kernels
IEEE fp32/NaN/signed-zero semantics
PyTorch dtype promotion
broadcasting with complex shape constraints
reductions/softmax/matmul
randomness/dropout
atomics/races
```

## Difficulty Estimate

Rough effort:

```text
Tiny tensor model + add/sub/mul: 1-2 weeks
Triton elementwise execution model: 2-4 weeks
Coverage/mask/no-OOB proof library: 2-4 weeks
Broadcasting/strides: 1-2 months
Reductions/softmax: 1-3 months
Faithful IEEE fp32 semantics: hard, likely separate project
```

For a narrow paper/demo artifact, 4-8 weeks is plausible. Broad TritonBench
coverage would be many months.

## Suggested Role For Lean

Lean should not be the first high-throughput checker. Use it as:

```text
1. a formal specification of the accepted semantic IR,
2. a proof that generic kernel schemas are correct,
3. a machine-checked audit artifact for selected examples,
4. a way to validate the meaning of SMT proof obligations.
```

The practical architecture is:

```text
semantic IR -> SMT-LIB/Z3
  fast proof/counterexample search

semantic IR -> Lean
  trusted semantics and selected machine-checked proofs
```

## First Lean Milestone

Formalize the `add` false-positive example:

```text
Spec:
  out[i] = input[i] + alpha * other[i]

Generated naive kernel:
  out[i] = input[i] + other[i]
```

Lean deliverables:

```text
1. Tensor model over `Int`
2. `torchAdd`
3. `tritonAddNaive`
4. theorem: correct when alpha = 1
5. theorem: not correct for all alpha
6. concrete counterexample: n=1, input[0]=0, other[0]=1, alpha=2
```

