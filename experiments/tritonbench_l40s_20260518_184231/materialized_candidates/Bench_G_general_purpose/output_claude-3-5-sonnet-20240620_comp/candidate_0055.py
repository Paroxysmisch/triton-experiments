import torch
import triton
import triton.language as tl
import math

# Helper function to calculate optimal kernel settings
def calculate_settings(n_cols):
    # Find the next power of 2 for BLOCK_SIZE, limited by MAX_FUSED_SIZE
    MAX_FUSED_SIZE = 2048
    BLOCK_SIZE = min(MAX_FUSED_SIZE, 1 << (n_cols - 1).bit_length())
    # Adjust num_warps based on BLOCK_SIZE
    num_warps = 4 if BLOCK_SIZE < 512 else 8
    return BLOCK_SIZE, num_warps

# SiLU activation function
@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

# Forward kernel for SWiGLU
@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_row, stride_col,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Position of block
    pid = tl.program_id(0)
    # Column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Mask for valid memory accesses
    mask = col_offsets < n_cols
    
    # Pointers to current row
    row_start_a = a_ptr + pid * stride_row
    row_start_b = b_ptr + pid * stride_row
    row_start_c = c_ptr + pid * stride_row
    
    # Load data
    a = tl.load(row_start_a + col_offsets * stride_col, mask=mask)
    b = tl.load(row_start_b + col_offsets * stride_col, mask=mask)
    
    # Compute SWiGLU: b * silu(a)
    c = b * silu(a)
    
    # Store result
    tl.store(row_start_c + col_offsets * stride_col, c, mask=mask)

# Backward kernel for SWiGLU
@triton.jit
def _swiglu_backward_kernel(
    grad_output_ptr, a_ptr, b_ptr,
    grad_a_ptr, grad_b_ptr,
    stride_row, stride_col,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load data
    row_start = pid * stride_row
    grad_output = tl.load(grad_output_ptr + row_start + col_offsets * stride_col, mask=mask)
    a = tl.load(a_ptr + row_start + col_offsets * stride_col, mask=mask)
    b = tl.load(b_ptr + row_start + col_offsets * stride_col, mask=mask)
    
    # Recompute forward pass
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a
    
    # Compute gradients
    # grad_a = grad_output * b * (silu'(a)) = grad_output * b * (sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a)))
    grad_a = grad_output * b * (sig_a + silu_a * (1 - sig_a))
    # grad_b = grad_output * silu(a)
    grad_b = grad_output * silu_a
    
    # Store gradients
    tl.store(grad_a_ptr + row_start + col_offsets * stride_col, grad_a, mask=mask)
    tl.store(grad_b_ptr + row_start + col_offsets * stride_col, grad_b, mask=mask)

# Wrapper function for forward pass
def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_contiguous(), "Input tensor 'a' must be contiguous"
    assert b.is_contiguous(), "Input tensor 'b' must be contiguous"
    
    # Reshape inputs to 2D
    shape = a.shape
    a_2d = a.reshape(-1, shape[-1])
    b_2d = b.reshape(-1, shape[-1])
    
    # Prepare output
    c = torch.empty_like(a_2d)
    
    # Calculate kernel settings
    BLOCK_SIZE, num_warps = calculate_settings(a_2d.shape[1])
    
    # Launch kernel
    grid = (a_2d.shape[0],)
    _swiglu_forward_kernel[grid](
        a_2d, b_2d, c,
        a_2d.stride(0), a_2d.stride(1),
        a_2d.shape[0], a_2d.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return c.reshape(shape)

# Wrapper function for backward pass
def swiglu_backward(grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    assert grad_output.is_contiguous(), "grad_output must be contiguous"
    
    # Reshape inputs to 2D
    shape = grad_output.shape
    grad_output_2d = grad_output.reshape(-1, shape[-1])
    a_2d = a.reshape(-1, shape[-1])
    b_2d = b.reshape(-1, shape[-1])
    
    # Prepare gradient outputs
    grad_a = torch.empty_like(a_2d)
    grad_b = torch.empty_like(b_2d)
    
    # Calculate kernel settings
    BLOCK_SIZE, num_warps = calculate_settings(a_2d.shape[1])
    
    # Launch kernel
    grid = (a_2d.shape[0],)
    _swiglu_backward_kernel[grid](
        grad_output_2d, a_2d, b_2d,
        grad_a, grad_b,
        a_2d.stride(0), a_2d.stride(1),
        a_2d.shape[0], a_2d.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return grad_a.reshape(shape), grad_b.reshape(shape)
