= Strict I/O Stdout Bucket Classification

#set text(size: 10pt)
#set heading(numbering: "1.")
#set page(margin: 0.85in)

This separates cases where the original I/O oracle has no meaningful signal because both scripts print empty stdout from cases with non-empty stdout.

#table(
  columns: 2,
  [field], [value],
  [matched candidates], [`234`],
  [empty equal stdout], [`230`],
  [non-empty equal stdout], [`0`],
  [stdout differs], [`4`],
)

== Main Findings

- Empty stdout is the whole false-positive story in this L40S clean rerun: all `156` stdout false positives have `candidate_stdout == reference_stdout == ""`.
- There are no cases where stdout is non-empty, equal, and the generated implementation is still incorrect.
- There are `4` non-empty generated stdout cases, but they do not pass the original stdout oracle because the reference stdout is empty.

== `empty_equal_stdout`

#table(
  columns: 2,
  [field], [value],
  [total], [`230`],
  [repo stdout equal], [`230`],
  [stdout false positives], [`156`],
  [classification counts], [`{'not_runnable': 74, 'strict_incorrect': 156}`],
)

- `G` / `fused_recurrent_retention`: `not_runnable` x `4`
- `G` / `fused_recurrent_retention`: `strict_incorrect` x `11`
- `G` / `l2_norm_bwd`: `strict_incorrect` x `2`
- `G` / `triton_mul2`: `not_runnable` x `2`
- `G` / `triton_mul2`: `strict_incorrect` x `20`
- `T` / `broadcast_tensors`: `not_runnable` x `6`
- `T` / `broadcast_tensors`: `strict_incorrect` x `12`
- `T` / `det`: `not_runnable` x `1`
- `T` / `det`: `strict_incorrect` x `10`
- `T` / `dropout_sigmoid_linear`: `not_runnable` x `3`
- `T` / `dropout_sigmoid_linear`: `strict_incorrect` x `14`
- `T` / `elu_linear`: `not_runnable` x `4`
- `T` / `elu_linear`: `strict_incorrect` x `7`
- `T` / `fused_cross_entropy_log_softmax`: `not_runnable` x `6`
- `T` / `fused_cross_entropy_log_softmax`: `strict_incorrect` x `9`
- `T` / `fused_qr_solve`: `not_runnable` x `3`
- `T` / `fused_qr_solve`: `strict_incorrect` x `12`
- `T` / `fused_transformer_block`: `not_runnable` x `7`
- `T` / `fused_transformer_block`: `strict_incorrect` x `11`
- `T` / `invert_matrix_lu`: `not_runnable` x `11`
- `T` / `invert_matrix_lu`: `strict_incorrect` x `7`
- `T` / `lu.py`: `not_runnable` x `8`
- `T` / `lu.py`: `strict_incorrect` x `10`
- `T` / `sigmoid_adaptive_avg_pool2d`: `not_runnable` x `8`
- `T` / `sigmoid_adaptive_avg_pool2d`: `strict_incorrect` x `9`
- `T` / `sum_std`: `not_runnable` x `10`
- `T` / `sum_std`: `strict_incorrect` x `6`
- `T` / `tensordot_rsqrt`: `not_runnable` x `1`
- `T` / `tensordot_rsqrt`: `strict_incorrect` x `16`

== `stdout_diff`

#table(
  columns: 2,
  [field], [value],
  [total], [`4`],
  [repo stdout equal], [`0`],
  [stdout false positives], [`0`],
  [classification counts], [`{'strict_incorrect': 4}`],
)

- `G` / `triton_mul2`: `strict_incorrect` x `3`
- `T` / `elu_linear`: `strict_incorrect` x `1`

== Non-Empty Stdout Records

=== `G` / `triton_mul2`

#table(
  columns: 2,
  [field], [value],
  [candidate], [`Bench_G_general_purpose/Qwen2.5-72B-Instruct_simp/candidate_0148.py`],
  [classification], [`strict_incorrect`],
  [repo stdout equal], [`False`],
  [candidate stdout], [`"Original tensor: tensor([1., 2., 3., 4., 5.])\nDoubled tensor (new tensor): tensor([ 2.,  4.,  6.,  8., 10.], device='cuda:0')\nDoubled tensor (in place): tensor([1., 2., 3., 4., 5.])\n"`],
  [reference stdout], [`''`],
)

=== `G` / `triton_mul2`

#table(
  columns: 2,
  [field], [value],
  [candidate], [`Bench_G_general_purpose/Qwen2.5-72B-Instruct_simp_rag/candidate_0149.py`],
  [classification], [`strict_incorrect`],
  [repo stdout equal], [`False`],
  [candidate stdout], [`"Original tensor: tensor([1, 2, 3, 4, 5], device='cuda:0')\nResult of triton_mul2: tensor([ 2,  4,  6,  8, 10], device='cuda:0')\nResult of triton_mul2_inplace: tensor([ 2,  4,  6,  8, 10], device='cuda:0')\n"`],
  [reference stdout], [`''`],
)

=== `G` / `triton_mul2`

#table(
  columns: 2,
  [field], [value],
  [candidate], [`Bench_G_general_purpose/output_gpt-4o_simp/candidate_0148.py`],
  [classification], [`strict_incorrect`],
  [repo stdout equal], [`False`],
  [candidate stdout], [`"Output (new tensor): tensor([2., 4., 6., 8.], device='cuda:0')\nOutput (in place): tensor([2., 4., 6., 8.], device='cuda:0')\n"`],
  [reference stdout], [`''`],
)

=== `T` / `elu_linear`

#table(
  columns: 2,
  [field], [value],
  [candidate], [`Bench_T_general_purpose/output_DeepSeek-R1_rag/def_0146.py`],
  [classification], [`strict_incorrect`],
  [repo stdout equal], [`False`],
  [candidate stdout], [`'Max difference: 80.40392303466797\n'`],
  [reference stdout], [`''`],
)

