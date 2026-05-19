import torch
import triton
import triton.language as tl
from .utils import calculate_settings

@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr, b_ptr, c_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    a_ptr += pid * n_cols
    b_ptr += pid * n_cols
    c_ptr += pid * n_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    a = tl.load(a_ptr + cols, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0).to(tl.float32)

    # GeGLU activation function: 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    tanh_arg = sqrt_2_over_pi * (a + 0.044715 * tl.pow(a, 3))
    tanh_result = tl.tanh(tanh_arg)
    geglu_a = 0.5 * a * (1 + tanh_result)

    c = geglu_a * b
    tl.store(c_ptr + cols, c, mask=mask)

def geglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = a.shape
    c = torch.empty((n_rows, n_cols), dtype=a.dtype, device=a.device)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    _geglu_tanh_forward_kernel[grid](a, b, c, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return c

@triton.jit
def _geglu_tanh_backward_kernel(
    a_ptr, b_ptr, dc_ptr, a_grad_ptr, b_grad_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    a_ptr += pid * n_cols
    b_ptr += pid * n_cols
    dc_ptr += pid * n_cols
    a_grad_ptr += pid * n_cols
    b_grad_ptr += pid * n_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    a = tl.load(a_ptr + cols, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0).to(tl.float32)
    dc = tl.load(dc_ptr + cols, mask=mask, other=0).to(tl.float32)

    # recomputation to avoid storing extra things
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    tanh_arg = sqrt_2_over_pi * (a + 0.044715 * tl.pow(a, 3))
    tanh_result = tl.tanh(tanh_arg)
    geglu_a = 0.5 * a * (1 + tanh_result)

    b_grad = dc * geglu_a
    tl.store(b_grad_ptr + cols, b_grad, mask=mask)

    a_grad = dc * b * (0.5 * (1 + tanh_result) * (1 + sqrt_2_over_pi * (3 * a * tl.pow(0.044715, 2) + 1) * tl.cosh(tanh_arg)))

    tl.store(a_grad_ptr + cols, a_grad, mask=mask)

def geglu_backward(a: torch.Tensor, b: torch.Tensor, dc: torch.Tensor) -> tuple:
    n_rows, n_cols = a.shape
    a_grad = torch.empty((n_rows, n_cols), dtype=a.dtype, device=a.device)
    b_grad = torch.empty((n_rows, n_cols), dtype=a.dtype, device=a.device)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_SIZE"]),)
    _geglu_tanh_backward_kernel[grid](
        a, b, dc, a_grad, b_grad, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return a_grad, b_grad
