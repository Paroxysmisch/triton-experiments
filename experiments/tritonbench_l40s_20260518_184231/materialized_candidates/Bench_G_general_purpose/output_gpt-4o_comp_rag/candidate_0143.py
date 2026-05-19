import triton
import triton.language as tl
import torch

# Triton kernel for approximate forward GEGLU operation using tanh-based approximation
@triton.jit
def _geglu_tanh_forward_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    s = 0.7978845608028654  # math.sqrt(2 / math.pi)
    a_row = tl.load(a_ptr + offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + offsets, mask=mask, other=0)

    # GEGLU activation function using tanh-based approximation
    f_row = 0.5 * a_row * (
        tl.math.tanh(s * a_row * (1.0 + 0.044715 * a_row * a_row)) + 1.0
    )
    f_row = f_row.to(b_row.dtype)
    c_row = f_row * b_row

    tl.store(c_ptr + offsets, c_row, mask=mask)

# Python function that wraps the approximate forward kernel
def geglu_forward(a, b):
    batch, seq_len, hd = a.shape
    n_elements = a.numel()
    c = torch.empty((batch, seq_len, hd), dtype=a.dtype, device="cuda")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _geglu_tanh_forward_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=128)
    return c

# Triton kernel for approximate backward GEGLU operation
@triton.jit
def _geglu_tanh_backward_kernel(dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dc_row = tl.load(dc_ptr + offsets, mask=mask, other=0)
    a_row = tl.load(a_ptr + offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + offsets, mask=mask, other=0)

    s = 0.7978845608028654  # math.sqrt(2 / math.pi)
    a = s * a_row
    b = a * 0.044715 * a_row * a_row
    T = 1.0 + tl.math.tanh(a + b)
    T2 = 0.5 * T
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
    df_da = T2 + Q2

    f_row = T2 * a_row
    f_row = f_row.to(dc_row.dtype)
    c_row = f_row * b_row
    df_row = dc_row * f_row
    db_row = dc_row * b_row

    da_row = db_row.to(tl.float32) * df_da
    da_row = da_row.to(dc_row.dtype)

    tl.store(da_ptr + offsets, df_row, mask=mask)
    tl.store(db_ptr + offsets, da_row, mask=mask)

# Python function that wraps the approximate backward kernel
def geglu_backward(dc, a, b):
    batch, seq_len, hd = a.shape
    n_elements = a.numel()
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _geglu_tanh_backward_kernel[grid](dc, a, b, da, db, n_elements, BLOCK_SIZE=128)
    return da, db
