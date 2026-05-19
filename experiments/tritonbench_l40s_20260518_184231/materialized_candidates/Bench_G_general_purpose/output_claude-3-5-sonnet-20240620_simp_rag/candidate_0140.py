import triton
import triton.language as tl
import torch

@triton.jit
def _geglu_tanh_forward_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # GELU approximation using tanh
    x = 0.7978845608028654 * (a + 0.044715 * a * a * a)
    y = a * 0.5 * (1.0 + tl.math.tanh(x))

    # GEGLU
    c = y * b

    tl.store(c_ptr + offsets, c, mask=mask)

@triton.jit
def _geglu_tanh_backward_kernel(
    grad_output_ptr, a_ptr, b_ptr, grad_a_ptr, grad_b_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # GELU derivative approximation
    x = 0.7978845608028654 * (a + 0.044715 * a * a * a)
    tanh_x = tl.math.tanh(x)
    sech_x_squared = 1 - tanh_x * tanh_x
    gelu_derivative = 0.5 * (1 + tanh_x + a * sech_x_squared * (0.7978845608028654 * (1 + 0.134145 * a * a)))

    # GEGLU backward
    grad_a = grad_output * b * gelu_derivative
    grad_b = grad_output * (0.5 * a * (1 + tanh_x))

    tl.store(grad_a_ptr + offsets, grad_a, mask=mask)
    tl.store(grad_b_ptr + offsets, grad_b, mask=mask)

def calculate_settings(n_elements):
    BLOCK_SIZE = 128
    num_warps = 4
    return BLOCK_SIZE, num_warps

def geglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    assert a.shape == b.shape, "Input tensors must have the same shape"
    
    batch, seq_len, hidden_dim = a.shape
    a = a.reshape(-1, hidden_dim)
    b = b.reshape(-1, hidden_dim)
    
    n_elements = a.numel()
    c = torch.empty_like(a)
    
    BLOCK_SIZE, num_warps = calculate_settings(n_elements)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _geglu_tanh_forward_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return c.reshape(batch, seq_len, hidden_dim)

def geglu_backward(grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert grad_output.is_cuda and a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    assert grad_output.shape == a.shape == b.shape, "Input tensors must have the same shape"
    
    batch, seq_len, hidden_dim = a.shape
    grad_output = grad_output.reshape(-1, hidden_dim)
    a = a.reshape(-1, hidden_dim)
    b = b.reshape(-1, hidden_dim)
    
    n_elements = a.numel()
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    
    BLOCK_SIZE, num_warps = calculate_settings(n_elements)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _geglu_tanh_backward_kernel[grid](
        grad_output, a, b, grad_a, grad_b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return grad_a.reshape(batch, seq_len, hidden_dim), grad_b.reshape(batch, seq_len, hidden_dim)
