import triton
import triton.language as tl
import torch

# Triton kernel for approximate forward GEGLU using tanh
@triton.jit
def _geglu_tanh_forward_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    s = 0.7978845608028654  # math.sqrt(2 / math.pi)
    a_row = tl.load(a + offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + offsets, mask=mask, other=0)

    f_row = 0.5 * a_row * (
        tl.math.tanh(s * a_row * (1.0 + 0.044715 * a_row * a_row)) + 1.0
    )
    f_row = f_row.to(b_row.dtype)
    c_row = f_row * b_row

    tl.store(c + offsets, c_row, mask=mask)

# Python function that wraps the approximate forward kernel
def geglu_tanh_forward(a, b):
    batch, seq_len, hd = a.shape
    n_elements = a.numel()
    out = torch.empty((batch, seq_len, hd), dtype=a.dtype, device="cuda")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _geglu_tanh_forward_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=128)
    return out

# Triton kernel for approximate backward GEGLU using tanh
@triton.jit
def _geglu_tanh_backward_kernel(grad_output, a, b, grad_a, grad_b, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out_row = tl.load(grad_output + offsets, mask=mask, other=0)
    a_row = tl.load(a + offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + offsets, mask=mask, other=0)

    s = 0.7978845608028654  # math.sqrt(2 / math.pi)
    a_s = s * a_row
    a_s_cubed = a_s * 0.044715 * a_row * a_row
    tanh_term = tl.math.tanh(a_s + a_s_cubed)
    T = 1.0 + tanh_term
    T2 = 0.5 * T
    Q2 = -T2 * (T - 2.0) * (a_s + 3.0 * a_s_cubed)
    df_da = T2 + Q2

    f_row = T2 * a_row
    f_row = f_row.to(grad_out_row.dtype)
    h_row = f_row * b_row
    df_row = grad_out_row * f_row
    dg_row = grad_out_row * b_row

    da_row = dg_row.to(tl.float32) * df_da
    da_row = da_row.to(grad_out_row.dtype)

    tl.store(grad_a + offsets, df_row, mask=mask)
    tl.store(grad_b + offsets, da_row, mask=mask)

# Python function that wraps the approximate backward kernel
def geglu_tanh_backward(grad_output, a, b):
    batch, seq_len, hd = a.shape
    n_elements = a.numel()
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _geglu_tanh_backward_kernel[grid](grad_output, a, b, grad_a, grad_b, n_elements, BLOCK_SIZE=128)
    return grad_a, grad_b
