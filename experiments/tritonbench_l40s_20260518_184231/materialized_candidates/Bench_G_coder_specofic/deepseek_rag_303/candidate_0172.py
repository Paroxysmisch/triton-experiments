import torch
import triton
import triton.language as tl
from triton import autotune, heuristics

# --- Triton Kernels ---
@triton.jit
def log_softmax_kernel(
    output_ptr,  # Pointer to the output array in device memory
    input_ptr,   # Pointer to the input array in device memory
    M,           # Number of rows in X
    N,           # Number of columns in X
    x_row_stride,  # Stride necessary to jump 1 row
    x_col_stride,  # Stride necessary to jump 1 column
    max_cols_ptr,  # Pointer to an array storing row-wise maxima
    log_sum_exp_ptr,  # Pointer to an array storing log(sum(exp(col)))
    BLOCK_M: tl.constexpr,  # Block size in rows
    BLOCK_N: tl.constexpr,  # Block size in cols
):
    # The log_softmax kernel consists of three parts:
    #   1. Compute the maxima and row-wise stability
    #   2. Compute exp(X - max)
    #   3. Compute log(sum(exp(X - max))) and store result + max which gives log_softmax(X)
    #
    # To achieve this, we map blocks of BLOCK_M rows and BLOCK_N cols across all threads.

    # Part 1: Compute max (col-wise)
    # -------------------------------
    # Each thread block corresponds to a column in X.
    # The block computes the max of its column.
    # The thread IDs represent row indices.
    row_start = tl.program_id(0) * BLOCK_M
    col_idx = tl.program_id(1)
    row_ids = row_start + tl.arange(0, BLOCK_M)
    x_row_ptrs = (input_ptr + col_idx * x_col_stride) + row_ids * x_row_stride
    mask = row_ids < M

    # Initialize max to the first number and then find maximum
    x = tl.load(x_row_ptrs, mask, other=-float("inf")).to(tl.float32)
    _max = tl.max(x, axis=0)

    # Write out max for this col into max_cols. 
    max_cols_ptrs = max_cols_ptr + col_idx * x_col_stride
    tl.store(max_cols_ptrs, _max, mask=True)

    tl.debug_barrier()

    # Part 2: Compute exp(X - max)
    # ----------------------------
    # Load X - max and exp it (but don't write to X)
    # Then, write result back to X: exp(X - max)
    row_start = tl.program_id(0) * BLOCK_M
    col_idx = tl.program_id(1)
    row_ids = row_start + tl.arange(0, BLOCK_M)
    x_row_ptrs = (input_ptr + col_idx * x_col_stride) + row_ids * x_row_stride

    x_minus_max = tl.load(x_row_ptrs, mask, other=-float(
        "inf")).to(tl.float32) - _max  # Load X - max
    exp_x_minus_max = tl.exp(x_minus_max)

    # Write out the result.
    tl.store(x_row_ptrs, exp_x_minus_add, mask=mask)

    tl.debug_barrier()

    # Part 3: Compute log(sum(exp(X - max))) and store result + max
    # -----------------------------------------------------------
    # Do a reduction sum on the row (this could be done clever but will
    # involve complex memory offsets so we'll simplify).
    row_start = tl.program_id(0) * BLOCK_M
    col_idx = tl.program_id(1)
    row_ids = row_start + tl.arange(0, BLOCK_M)
    out_row_ptrs = (output_ptr + col_idx * x_col_stride) + row_ids * x_row_stride

    # reset row to the first number and then find sum
    x = tl.load(x_row_ptrs, mask, other=0.0).to(tl.float32)
    log_sum_exp = tl.log(tl.sum(x, axis=0) + 1e-12) + _max

    tl.store(log_sum_exp_ptr + col_idx, log_sum_exp, mask=True)
    tl.store(out_row_ptrs, exp_x_minus_add / tl.exp(log_sum_exp), mask=mask) 


@triton.jit
def log_softmax_backward_kernel(
    out_ptr,      # Pointer to the output array in device memory
    out_grad_ptr,  # Pointer to the output gradient array
    in_ptr,       # Pointer to the input array
    M, N,         # Dimensions of X
    x_row_stride,  # Stride necessary to jump 1 row
    x_col_stride,  # Stride necessary to jump 1 column
    max_cols_ptr,  # Pointer to an array storing row-max
    log_sum_exp,   # log(sum(exp(cols)))
    BLOCK_M: tl.constexpr,  # Block size in rows
    BLOCK_N: tl.constexpr,  # Block size in cols
):
    # Map blocks of BLOCK_M rows and BLOCK_N cols across all threads.
    row_start = tl.program_id(0) * BLOCK_M
    col_idx = tl.program_id(1)
    row_ids = row_start + tl.arange(0, BLOCK_M)
    col_ids = tl.arange(0, BLOCK_N) + col_idx * x_col_stride

    output_ptrs = (out_ptr + row_ids[:, None] * x_row_stride +
                   col_ids[None, :] * x_col_stride)
    out_grad_ptrs = (out_grad_ptr + row_ids[:,
                                            None] * x_row_stride + col_ids[None, :] * x_col_stride)
    mask = (row_ids < M)[:, None] & (col_ids < N)[None, :]

    # Load outputs and gradients
    out = tl.load(output_ptrs, mask=mask).to(tl.float32)
    out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)

    # Apply the chain rule
    # grad[i] = out[i] * (sum[j] - out[j] * out[i])
    numerator = tl.sum(out_grad * out, axis=1)
    denominator = tl.exp(log_sum_exp)
    x = numerator / denominator[:, None] - out * (numerator /
                                                 denominator[:, None])  # "out[j] * out[i]" term
    x *= out  # "sum[j] - ..." term

    # Write the result
    tl.store(out_grad_ptrs, x.to(tl.float32), mask=mask)


class LogSoftmax(torch.autograd.Function):
    @staticmethod
    @autotune(
        configs=[
            triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w)
            for m in [16, 32, 64]
            for n in [32, 64, 128]
            for w in [4, 8]
        ],
        key=[],
        heuristics=[heuristics.first_trial_fails(2), heuristics.exhaustive()]
    )
    def forward
