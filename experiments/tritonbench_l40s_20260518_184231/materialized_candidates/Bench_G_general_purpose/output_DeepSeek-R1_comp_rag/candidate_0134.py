import triton
import triton.language as tl
import torch

@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    for col_base in tl.range(0, n_cols, BLOCK_SIZE):
        cols = col_base + col_offsets
        mask = cols < n_cols

        a = tl.load(a_ptr + row_idx * n_cols + cols, mask=mask, other=0).to(tl.float32)
        b = tl.load(b_ptr + row_idx * n_cols + cols, mask=mask, other=0)

        # Compute GEGLU using tanh approximation
        s = 0.7978845608028654  # sqrt(2/pi)
        a_cubed = a * a * a
        inner = s * (a + 0.044715 * a_cubed)
        tanh_val = tl.tanh(inner)
        gelu = 0.5 * a * (1.0 + tanh_val)
        c = gelu * b

        tl.store(c_ptr + row_idx * n_cols + cols, c, mask=mask)

def geglu_forward(a: torch.Tensor, b: torch.Tensor):
    a_2d = a.reshape(-1, a.shape[-1])
    b_2d = b.reshape(-1, b.shape[-1])
    n_rows, n_cols = a_2d.shape

    c = torch.empty_like(a_2d)
    BLOCK_SIZE = 128
    num_warps = 4

    grid = (n_rows,)
    _geglu_tanh_forward_kernel[grid](
        a_2d, b_2d, c,
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return c.view_as(a)

@triton.jit
def _geglu_tanh_backward_kernel(
    dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    for col_base in tl.range(0, n_cols, BLOCK_SIZE):
        cols = col_base + col_offsets
        mask = cols < n_cols

        dc = tl.load(dc_ptr + row_idx * n_cols + cols, mask=mask, other=0)
        a = tl.load(a_ptr + row_idx * n_cols + cols, mask=mask, other=0).to(tl.float32)
        b = tl.load(b_ptr + row_idx * n_cols + cols, mask=mask, other=0)

        # Recompute GELU and compute derivatives
        s = 0.7978845608028654
        a_part = s * a
        b_part = a_part * 0.044715 * a * a
        u = a_part + b_part
        tanh_u = tl.tanh(u)
        T = 1.0 + tanh_u
        gelu = 0.5 * a * T

        # Compute derivative of GELU
        Q_part = a_part + 3.0 * b_part
        T_minus_2 = T - 2.0
        Q2 = -0.5 * T * T_minus_2 * Q_part
        df_de = 0.5 * T + Q2

        # Calculate gradients
        da_grad = (dc * b) * df_de
        db_grad = dc * gelu

        tl.store(da_ptr + row_idx * n_cols + cols, da_grad.to(a_ptr.dtype.element_ty), mask=mask)
        tl.store(db_ptr + row_idx * n_cols + cols, db_grad.to(b_ptr.dtype.element_ty), mask=mask)

def geglu_backward(dc: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    dc_2d = dc.reshape(-1, dc.shape[-1])
    a_2d = a.reshape(-1, a.shape[-1])
    b_2d = b.reshape(-1, b.shape[-1])
    n_rows, n_cols = a_2d.shape

    da = torch.empty_like(a_2d)
    db = torch.empty_like(b_2d)
    BLOCK_SIZE = 128
    num_warps = 4

    grid = (n_rows,)
    _geglu_tanh_backward_kernel[grid](
        dc_2d, a_2d, b_2d, da, db,
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return da.view_as(a), db.view_as(b)
