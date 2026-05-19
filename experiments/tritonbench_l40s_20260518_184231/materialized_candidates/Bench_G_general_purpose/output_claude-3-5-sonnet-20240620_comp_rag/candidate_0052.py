import triton
import triton.language as tl
import torch

@triton.jit
def silu(x):
    """SiLU activation function: x * sigmoid(x)"""
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Forward kernel for SWiGLU operation.
    Computes c = b * silu(a) element-wise.
    """
    # Get the program ID and compute the row to process
    pid = tl.program_id(0)
    
    # Offset the pointers to the start of the row
    a_ptr = a_ptr + pid * stride
    b_ptr = b_ptr + pid * stride
    c_ptr = c_ptr + pid * stride
    
    # Create a range for the columns and a mask for bounds checking
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # Load data with masking
    a = tl.load(a_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    
    # Compute SWiGLU: c = b * silu(a)
    c = b * silu(a)
    
    # Store the result
    tl.store(c_ptr + cols, c, mask=mask)

@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
    stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Backward kernel for SWiGLU operation.
    Computes gradients for a and b using recomputation.
    """
    # Get the program ID and compute the row to process
    pid = tl.program_id(0)
    
    # Offset the pointers to the start of the row
    dc_ptr = dc_ptr + pid * stride
    a_ptr = a_ptr + pid * stride
    b_ptr = b_ptr + pid * stride
    da_ptr = da_ptr + pid * stride
    db_ptr = db_ptr + pid * stride
    
    # Create a range for the columns and a mask for bounds checking
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # Load data with masking
    dc = tl.load(dc_ptr + cols, mask=mask, other=0.0)
    a = tl.load(a_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    
    # Recompute intermediate values
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a
    
    # Compute gradients
    # db = dc * silu(a)
    db = dc * silu_a
    # da = dc * b * (silu(a) * (1 - sigmoid(a)) + sigmoid(a))
    da = dc * b * (silu_a * (1 - sig_a) + sig_a)
    
    # Store the gradients
    tl.store(da_ptr + cols, da, mask=mask)
    tl.store(db_ptr + cols, db, mask=mask)

def calculate_settings(n_cols):
    """Calculate optimal block size and number of warps."""
    # Maximum size that can be fused
    MAX_FUSED_SIZE = 65536
    
    # Calculate block size as power of 2
    block_size = min(MAX_FUSED_SIZE, 1 << (n_cols - 1).bit_length())
    
    # Calculate number of warps based on block size
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    elif block_size >= 4096:
        num_warps = 16
        
    return block_size, num_warps

def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Forward pass of SWiGLU.
    Args:
        a: Input tensor
        b: Input tensor (same shape as a)
    Returns:
        c: Output tensor, b * silu(a)
    """
    # Save original shape and reshape to 2D
    orig_shape = a.shape
    n_cols = orig_shape[-1]
    a = a.reshape(-1, n_cols)
    b = b.reshape(-1, n_cols)
    
    # Create output tensor
    c = torch.empty_like(a)
    n_rows = a.shape[0]
    
    # Calculate kernel settings
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    
    # Launch kernel
    _swiglu_forward_kernel[(n_rows,)](
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        a.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return c.reshape(orig_shape)

def swiglu_backward(dc: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Backward pass of SWiGLU.
    Args:
        dc: Gradient of the loss with respect to the output
        a: Input tensor from forward pass
        b: Input tensor from forward pass
    Returns:
        da: Gradient with respect to a
        db: Gradient with respect to b
    """
    # Save original shape and reshape to 2D
    orig_shape = dc.shape
    n_cols = orig_shape[-1]
    dc = dc.reshape(-1, n_cols)
    a = a.reshape(-1, n_cols)
    b = b.reshape(-1, n_cols)
    
    # Create gradient tensors
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    n_rows = dc.shape[0]
    
    # Calculate kernel settings
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    
    # Launch kernel
    _swiglu_backward_kernel[(n_rows,)](
        dc.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        da.data_ptr(),
        db.data_ptr(),
        dc.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return da.reshape(orig_shape), db.reshape(orig_shape)
