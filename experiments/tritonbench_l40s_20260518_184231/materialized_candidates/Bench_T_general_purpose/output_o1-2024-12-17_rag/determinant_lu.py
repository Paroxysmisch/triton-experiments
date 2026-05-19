import triton
import triton.language as tl
import torch

# -----------------------------------------------------
# Triton kernel: performs one step of LU factorization
# at pivot row "pivot_idx". This kernel assumes that
# A has shape [batch_size, n, n] in row-major format.
# If pivot=True, row swapping will be performed
# according to the pivot row stored in pivot_rows tensor.
# -----------------------------------------------------
@triton.jit
def lu_factor_step_kernel(
    A_ptr,               # float* (batch of matrices)
    pivot_rows_ptr,      # int* (batch of pivot row indices, if pivot=True)
    batch_idx,           # Current batch index in [0..batch_size)
    pivot_idx,           # Current pivot index in [0..n)
    n,
    stride_batch,        # Stride in elements to jump from one matrix to the next
    stride_row,          # Stride in elements to jump from one row to the next
    pivot_flag,          # 1 if pivot=True, else 0
    BLOCK_SIZE: tl.constexpr
):
    # Each program processes columns in parallel
    # row offset for the pivot row
    pivot_row_offset = batch_idx * stride_batch + pivot_idx * stride_row
    # If pivot=True, swap the pivot row with the row that has maximum pivot element
    # If pivot_rows_ptr is not None, read which row to swap with
    if pivot_flag == 1:
        swap_row = tl.load(pivot_rows_ptr + batch_idx * n + pivot_idx)
        # if swap_row != pivot_idx, we need to swap the two rows
        # let thread c = col index
        c_seq = tl.arange(0, BLOCK_SIZE)
        c_mask = c_seq < n
        base_offset = batch_idx * stride_batch
        pivot_row_vals_ptr = A_ptr + base_offset + pivot_idx * stride_row + c_seq
        swap_row_vals_ptr = A_ptr + base_offset + swap_row * stride_row + c_seq
        # load
        pivot_data = tl.load(pivot_row_vals_ptr, mask=c_mask)
        swap_data = tl.load(swap_row_vals_ptr, mask=c_mask)
        # swap
        tl.store(pivot_row_vals_ptr,
