import triton
import triton.language as tl
import torch
import math

# Constants
SQRT_2_PI = math.sqrt(2 / math.pi)
COEFF = 0.044715

@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_a_row, stride_b_row, stride_c_row,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute offsets
    offs_m = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, n_cols)
    
    # Create mask
    mask = offs_m < n_rows
    
    # Pointers for the current row
    a_ptrs = a_ptr + offs_m[:, None] * stride_a_row + offs_n[None, :]
    b_ptrs = b_ptr + offs_m[:, None] * stride_b_row + offs_n[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_c_row + offs_n[None, :]
    
    # Load inputs
    a = tl.load(a_ptrs, mask=mask[:, None], other=0.0)
    b = tl.load(b_ptrs, mask=mask[:, None], other=0.0)
    
    # Compute GELU approximation using tanh
    inner = SQRT_2_PI * (a + COEFF * a * a * a)
    gelu = 0.5 * a * (1.0 + tl.math.tanh(inner))
    
    # Compute GEGLU
    c = gelu * b
    
    # Store result
    tl.store(c_ptrs, c, mask=mask[:, None])

@triton.jit
def _geglu_tanh_backward_kernel(
    grad_ptr, a_ptr, b_ptr,
    grad_a_ptr, grad_b_ptr,
    stride_row,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute offsets
    offs_m = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, n_cols)
    
    # Create mask
    mask = offs_m < n_rows
    
    # Pointers for the current row
    grad_ptrs = grad_ptr + offs_m[:, None] * stride_row + offs_n[None, :]
    a_ptrs = a_ptr + offs_m[:, None] * stride_row + offs_n[None, :]
    b_ptrs = b_ptr + offs_m[:, None] * stride_row + offs_n[None, :]
    grad_a_ptrs = grad_a_ptr + offs_m[:, None] * stride_row + offs_n[None, :]
    grad_b_ptrs = grad_b_ptr + offs_m[:, None] * stride_row + offs_n[None, :]
    
    # Load inputs
    grad = tl.load(grad_ptrs, mask=mask[:, None], other=0.0)
    a = tl.load(a_ptrs, mask=mask[:, None], other=0.0)
    b = tl.load(b_ptrs, mask=mask[:, None], other=0.0)
    
    # Recompute forward pass intermediates
    inner = SQRT_2_PI * (a + COEFF * a * a * a)
    tanh_inner = tl.math.tanh(inner)
    gelu = 0.5 * a * (1.0 + tanh_inner)
    
    # Compute gradients
    dtanh = 1.0 - tanh_inner * tanh_inner
    dgelu = 0.5 * (1.0 + tanh_inner + a * dtanh * SQRT_2_PI * (1.0 + 3.0 * COEFF * a * a))
    
    grad_a = grad * (dgelu * b)
    grad_b = grad * gelu
    
    # Store gradients
    tl.store(grad_a_ptrs, grad_a, mask=mask[:, None])
    tl.store(grad_b_ptrs, grad_b, mask=mask[:, None])

def calculate_settings(n_cols):
    # Simple heuristic for block size and num_warps
    BLOCK_SIZE = 128
    num_warps = 4
    return BLOCK_SIZE, num_warps

def geglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    assert a.dim() >= 2
    
    # Reshape inputs to 2D
    original_shape = a.shape
    n_rows = a.numel() // a.shape[-1]
    n_cols = a.shape[-1]
    a_2d = a.view(n_rows, n_cols)
    b_2d = b.view(n_rows, n_cols)
    
    # Initialize output
    c = torch.empty_like(a_2d)
    
    # Calculate kernel settings
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    
    # Launch kernel
    grid = (triton.cdiv(n_rows, BLOCK_SIZE),)
    _geglu_tanh_forward_kernel[grid](
        a_2d, b_2d, c,
        a_2d.stride(0), b_2d.stride(0), c.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return c.view(original_shape)

def geglu_backward(grad: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert grad.is_cuda and a.is_cuda and b.is_cuda
    assert grad.shape == a.shape == b.shape
    
    # Reshape inputs to 2D
    n_rows = grad.numel() // grad.shape[-1]
    n_cols = grad.shape[-1]
    grad_2d = grad.view(n_rows, n_cols)
    a_2d = a.view(n_rows, n_cols)
    b_2d = b.view(n_rows, n_cols)
    
    # Initialize gradients
    grad_a = torch.empty_like(a_2d)
    grad_b = torch.empty_like(b_2d)
    
    # Calculate kernel settings
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    
    # Launch kernel
    grid = (triton.cdiv(n_rows, BLOCK_SIZE),)
    _geglu_tanh_backward_kernel[grid](
        grad_2d, a_2d, b_2d,
        grad_a, grad_b,
        grad_2d.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return grad_a.view_as(a), grad_b.view_as(b)
