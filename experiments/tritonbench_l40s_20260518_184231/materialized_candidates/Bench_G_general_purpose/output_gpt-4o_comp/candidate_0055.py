import triton
import triton.language as tl
import torch

# Helper function to calculate block size and number of warps
def calculate_settings(num_cols):
    MAX_FUSED_SIZE = 128
    BLOCK_SIZE = min(2**((num_cols - 1).bit_length()), MAX_FUSED_SIZE)
    num_warps = min(max(1, BLOCK_SIZE // 32), 8)
    return BLOCK_SIZE, num_warps

# SiLU activation function
@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

# Forward kernel
@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, num_rows, num_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    col_start = tl.arange(0, BLOCK_SIZE)
    
    a_ptrs = a_ptr + row_start * num_cols + col_start
    b_ptrs = b_ptr + row_start * num_cols + col_start
    c_ptrs = c_ptr + row_start * num_cols + col_start

    a = tl.load(a_ptrs, mask=col_start < num_cols)
    b = tl.load(b_ptrs, mask=col_start < num_cols)
    
    # Apply SiLU to a
    a_silu = silu(a)
    
    # Element-wise multiply
    c = a_silu * b
    
    tl.store(c_ptrs, c, mask=col_start < num_cols)

# Backward kernel
@triton.jit
def _swiglu_backward_kernel(a_ptr, b_ptr, grad_out_ptr, grad_a_ptr, grad_b_ptr, num_rows, num_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    col_start = tl.arange(0, BLOCK_SIZE)
    
    a_ptrs = a_ptr + row_start * num_cols + col_start
    b_ptrs = b_ptr + row_start * num_cols + col_start
    grad_out_ptrs = grad_out_ptr + row_start * num_cols + col_start
    grad_a_ptrs = grad_a_ptr + row_start * num_cols + col_start
    grad_b_ptrs = grad_b_ptr + row_start * num_cols + col_start

    a = tl.load(a_ptrs, mask=col_start < num_cols)
    b = tl.load(b_ptrs, mask=col_start < num_cols)
    grad_out = tl.load(grad_out_ptrs, mask=col_start < num_cols)
    
    # Recompute SiLU and its derivative
    a_silu = silu(a)
    a_silu_prime = tl.sigmoid(a) * (1 + a * (1 - tl.sigmoid(a)))
    
    # Compute gradients
    grad_a = grad_out * b * a_silu_prime
    grad_b = grad_out * a_silu
    
    tl.store(grad_a_ptrs, grad_a, mask=col_start < num_cols)
    tl.store(grad_b_ptrs, grad_b, mask=col_start < num_cols)

# Forward wrapper
def swiglu_forward(a, b):
    assert a.shape == b.shape, "Input tensors must have the same shape"
    num_rows, num_cols = a.shape
    BLOCK_SIZE, num_warps = calculate_settings(num_cols)
    
    c = torch.empty_like(a)
    
    grid = (num_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    _swiglu_forward_kernel[grid](a, b, c, num_rows, num_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    
    return c

# Backward wrapper
def swiglu_backward(a, b, grad_out):
    assert a.shape == b.shape == grad_out.shape, "All tensors must have the same shape"
    num_rows, num_cols = a.shape
    BLOCK_SIZE, num_warps = calculate_settings(num_cols)
    
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    
    grid = (num_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    _swiglu_backward_kernel[grid](a, b, grad_out, grad_a, grad_b, num_rows, num_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    
    return grad_a, grad_b
