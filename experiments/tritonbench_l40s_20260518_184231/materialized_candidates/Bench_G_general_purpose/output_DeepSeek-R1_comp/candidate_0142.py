import torch
import triton
import triton.language as tl
import math

@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_cols,
    stride_a_row,
    stride_b_row,
    stride_c_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * stride_a_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + col_offsets
        curr_mask = mask & (offsets < n_cols)

        a_ptrs = a_ptr + row_start + offsets
        b_ptrs = b_ptr + row_start + offsets

        a = tl.load(a_ptrs, mask=curr_mask, other=0.0)
        b = tl.load(b_ptrs, mask=curr_mask, other=0.0)

        # Compute GELU(b) using tanh approximation
        sqrt_2_over_pi = tl.sqrt(2.0 / math.pi)
        approx = b + 0.044715 * b * b * b
        tanh_term = tl.tanh(sqrt_2_over_pi * approx)
        gelu_b = 0.5 * b * (1.0 + tanh_term)

        c = a * gelu_b

        c_ptrs = c_ptr + row_start + offsets
        tl.store(c_ptrs, c, mask=curr_mask)

@triton.jit
def _geglu_tanh_backward_kernel(
    a_ptr,
    b_ptr,
    dc_ptr,
    da_ptr,
    db_ptr,
    n_cols,
    stride_a_row,
    stride_b_row,
    stride_dc_row,
    stride_da_row,
    stride_db_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * stride_a_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + col_offsets
        curr_mask = mask & (offsets < n_cols)

        a_ptrs = a_ptr + row_start + offsets
        b_ptrs = b_ptr + row_start + offsets
        dc_ptrs = dc_ptr + row_start + offsets

        a = tl.load(a_ptrs, mask=curr_mask, other=0.0)
        b = tl.load(b_ptrs, mask=curr_mask, other=0.0)
        dc = tl.load(dc_ptrs, mask=curr_mask, other=0.0)

        # Recompute GELU(b) and its derivative
        sqrt_2_over_pi = tl.sqrt(2.0 / math.pi)
        approx = b + 0.044715 * b * b * b
        tanh_input = sqrt_2_over_pi * approx
        tanh_term = tl.tanh(tanh_input)
        gelu_b = 0.5 * b * (1.0 + tanh_term)

        # Derivative of GELU(b) with respect to b
        d_gelu_b = 0.5 * (1.0 + tanh_term) + 0.5 * b * (1.0 - tanh_term * tanh_term) * sqrt_2_over_pi * (1.0 + 0.134145 * b * b)

        da = dc * gelu_b
        db = dc * a * d_gelu_b

        da_ptrs = da_ptr + row_start + offsets
        db_ptrs = db_ptr + row_start + offsets

        tl.store(da_ptrs, da, mask=curr_mask)
        tl.store(db_ptrs, db, mask=curr_mask)

def geglu_forward(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, "a and b must have the same shape"
    a_2d = a.view(-1, a.size(-1))
    b_2d = b.view(-1, b.size(-1))
    n_rows, n_cols = a_2d.shape
    c = torch.empty_like(a_2d)
    BLOCK_SIZE = 128
    num_warps = 4
    grid = (n_rows,)
    _geglu_tanh_forward_kernel[grid](
        a_2d, b_2d, c, n_cols,
        a_2d.stride(0), b_2d.stride(0), c.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return c.view_as(a)

def geglu_backward(dc: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    a_2d = a.view(-1, a.size(-1))
    b_2d = b.view(-1, b.size(-1))
    dc_2d = dc.view(-1, dc.size(-1))
    n_rows, n_cols = a_2d.shape
    da = torch.empty_like(a_2d)
    db = torch.empty_like(b_2d)
    BLOCK_SIZE = 128
    num_warps = 4
    grid = (n_rows,)
    _geglu_tanh_backward_kernel[grid](
        a_2d, b_2d, dc_2d, da, db, n_cols,
        a_2d.stride(0), b_2d.stride(0), dc_2d.stride(0),
        da.stride(0), db.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return da.view_as(a), db.view_as(b)
