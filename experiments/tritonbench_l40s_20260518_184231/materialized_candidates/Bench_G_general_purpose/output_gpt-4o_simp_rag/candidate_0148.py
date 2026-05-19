import torch
import triton
import triton.language as tl

@triton.jit
def _softmax(Y, X, M, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, causal: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = X + row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    if causal:
        causal_mask = col_offsets <= tl.arange(0, n_cols)[None, :]
        mask = mask & causal_mask

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_ptrs = Y + row_idx * n_cols
    tl.store(output_ptrs, softmax_output, mask=mask)

@triton.jit
def _softmax_backward(dY, Y, dX, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = Y + row_idx * n_cols
    dy_start_ptr = dY + row_idx * n_cols
    dx_start_ptr = dX + row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    y_ptrs = row_start_ptr + col_offsets
    dy_ptrs = dy_start_ptr + col_offsets
    dx_ptrs = dx_start_ptr + col_offsets
    mask = col_offsets < n_cols

    y = tl.load(y_ptrs, mask=mask)
    dy = tl.load(dy_ptrs, mask=mask)
    sum_dy_y = tl.sum(dy * y, axis=0)
    dx = y * (dy - sum_dy_y)

    tl.store(dx_ptrs, dx, mask=mask)

def softmax(X, causal=False):
    n_rows, n_cols = X.shape[-2], X.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    Y = torch.empty_like(X)

    _softmax[(n_rows,)](Y, X, None, n_rows, n_cols, BLOCK_SIZE, causal)
    return Y

def softmax_backward(dY, Y):
    n_rows, n_cols = Y.shape[-2], Y.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    dX = torch.empty_like(Y)

    _softmax_backward[(n_rows,)](dY, Y, dX, n_rows, n_cols, BLOCK_SIZE)
    return dX
