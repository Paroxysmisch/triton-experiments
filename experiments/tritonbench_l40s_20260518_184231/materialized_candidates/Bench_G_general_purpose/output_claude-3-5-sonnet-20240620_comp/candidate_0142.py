import torch
import triton
import triton.language as tl
import math

# Constants for GELU approximation
SQRT_2_OVER_PI = math.sqrt(2 / math.pi)
COEFF = 0.044715

@triton.jit
def _geglu_tanh_forward_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    n_rows, n_cols,
    # Strides for accessing memory
    stride_row, stride_col,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute row index
    row_idx = pid
    
    # Handle bounds checking
    if row_idx >= n_rows:
        return
        
    # Compute memory offsets for current row
    a_row_off = row_idx * stride_row
    b_row_off = row_idx * stride_row
    c_row_off = row_idx * stride_row
    
    # Load columns in chunks of BLOCK_SIZE
    for col in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load values from a and b
        a_vals = tl.load(a_ptr + a_row_off + col_offsets * stride_col, mask=mask)
        b_vals = tl.load(b_ptr + b_row_off + col_offsets * stride_col, mask=mask)
        
        # Compute GELU using tanh approximation
        # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cube = a_vals * a_vals * a_vals
        inner = SQRT_2_OVER_PI * (a_vals + COEFF * cube)
        gelu = 0.5 * a_vals * (1.0 + tl.tanh(inner))
        
        # Multiply with gating values
        out = gelu * b_vals
        
        # Store result
        tl.store(c_ptr + c_row_off + col_offsets * stride_col, out, mask=mask)

@triton.jit
def _geglu_tanh_backward_kernel(
    # Pointers to matrices
    grad_ptr, a_ptr, b_ptr, 
    grad_a_ptr, grad_b_ptr,
    # Matrix dimensions
    n_rows, n_cols,
    # Strides for accessing memory
    stride_row, stride_col,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute row index
    row_idx = pid
    
    # Handle bounds checking
    if row_idx >= n_rows:
        return
        
    # Compute memory offsets for current row
    grad_row_off = row_idx * stride_row
    a_row_off = row_idx * stride_row
    b_row_off = row_idx * stride_row
    grad_a_row_off = row_idx * stride_row
    grad_b_row_off = row_idx * stride_row
    
    # Load columns in chunks of BLOCK_SIZE
    for col in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load values
        grad = tl.load(grad_ptr + grad_row_off + col_offsets * stride_col, mask=mask)
        a = tl.load(a_ptr + a_row_off + col_offsets * stride_col, mask=mask)
        b = tl.load(b_ptr + b_row_off + col_offsets * stride_col, mask=mask)
        
        # Recompute forward pass intermediates
        cube = a * a * a
        inner = SQRT_2_OVER_PI * (a + COEFF * cube)
        tanh_inner = tl.tanh(inner)
        gelu = 0.5 * a * (1.0 + tanh_inner)
        
        # Compute gradients
        # d_tanh = 1 - tanh^2
        d_tanh = 1.0 - tanh_inner * tanh_inner
        
        # d_gelu = 0.5 * (1 + tanh + a * d_tanh * sqrt(2/pi) * (1 + 3 * COEFF * a^2))
        d_gelu = 0.5 * (1.0 + tanh_inner + 
                        a * d_tanh * SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * a * a))
        
        # Compute gradients for a and b
        grad_a = grad * (d_gelu * b)
        grad_b = grad * gelu
        
        # Store gradients
        tl.store(grad_a_ptr + grad_a_row_off + col_offsets * stride_col, grad_a, mask=mask)
        tl.store(grad_b_ptr + grad_b_row_off + col_offsets * stride_col, grad_b, mask=mask)

def calculate_settings(n_elements):
    """Calculate BLOCK_SIZE and num_warps based on input size."""
    BLOCK_SIZE = 128
    num_warps = 4
    return BLOCK_SIZE, num_warps

class GEGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # Save tensors for backward pass
        ctx.save_for_backward(a, b)
        
        # Get input dimensions
        *batch_dims, n_cols = a.shape
        n_rows = a.numel() // n_cols
        
        # Reshape inputs to 2D
        a_2d = a.reshape(-1, n_cols)
        b_2d = b.reshape(-1, n_cols)
        
        # Initialize output
        c = torch.empty_like(a_2d)
        
        # Calculate launch configuration
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        
        # Launch kernel
        _geglu_tanh_forward_kernel[(n_rows,)](
            a_2d, b_2d, c,
            n_rows, n_cols,
            c.stride(0), c.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        # Reshape output back to original dimensions
        return c.reshape(*batch_dims, -1)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        a, b = ctx.saved_tensors
        
        # Get dimensions
        *batch_dims, n_cols = a.shape
        n_rows = a.numel() // n_cols
        
        # Reshape tensors to 2D
        grad_2d = grad_output.reshape(-1, n_cols)
        a_2d = a.reshape(-1, n_cols)
        b_2d = b.reshape(-1, n_cols)
        
        # Initialize gradient tensors
        grad_a = torch.empty_like(a_2d)
        grad_b = torch.empty_like(b_2d)
        
        # Calculate launch configuration
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        
        # Launch backward kernel
        _geglu_tanh_backward_kernel[(n_rows,)](
            grad_2d, a_2d, b_2d,
            grad_a, grad_b,
            n_rows, n_cols,
            grad_a.stride(0), grad_a.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        # Reshape gradients back to original dimensions
        return grad_a.reshape(*batch_dims, -1), grad_b.reshape(*batch_dims, -1)

def geglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Applies GEGLU activation function using tanh approximation.
    
    Args:
        a: Input tensor
        b: Gating tensor (same shape as a)
    
    Returns:
        Output tensor after applying GEGLU
    """
    return GEGLUFunction.apply(a, b)
