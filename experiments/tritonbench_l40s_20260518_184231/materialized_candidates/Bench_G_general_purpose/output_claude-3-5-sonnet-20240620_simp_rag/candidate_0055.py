import triton
import triton.language as tl
import torch

@triton.jit
def silu(x):
    """SiLU activation function: x * sigmoid(x)"""
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_forward_kernel(
    a_ptr,  # Pointer to first input tensor
    b_ptr,  # Pointer to second input tensor
    c_ptr,  # Pointer to output tensor
    stride, # Stride between rows
    n_cols: tl.constexpr,    # Number of columns (static)
    BLOCK_SIZE: tl.constexpr, # Block size for parallel processing
):
    # Get the program ID for the current thread block
    pid = tl.program_id(0)
    
    # Compute pointer offsets for this row
    offset = pid * stride
    a_ptr = a_ptr + offset
    b_ptr = b_ptr + offset
    c_ptr = c_ptr + offset
    
    # Create a range for accessing elements within the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load data for this row
    a = tl.load(a_ptr + col_offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute SWiGLU: SiLU(a) * b
    c = silu(a) * b
    
    # Store the result
    tl.store(c_ptr + col_offsets, c, mask=mask)

@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, # Pointer to output gradient
    a_ptr,  # Pointer to first input tensor
    b_ptr,  # Pointer to second input tensor
    da_ptr, # Pointer to gradient w.r.t. a
    db_ptr, # Pointer to gradient w.r.t. b
    stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute pointer offsets for this row
    offset = pid * stride
    dc_ptr = dc_ptr + offset
    a_ptr = a_ptr + offset
    b_ptr = b_ptr + offset
    da_ptr = da_ptr + offset
    db_ptr = db_ptr + offset
    
    # Create a range for accessing elements within the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load data
    dc = tl.load(dc_ptr + col_offsets, mask=mask, other=0.0)
    a = tl.load(a_ptr + col_offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute gradients
    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a
    
    # Gradient w.r.t b is dc * SiLU(a)
    db = dc * silu_a
    
    # Gradient w.r.t a is dc * b * d(SiLU)/da
    # where d(SiLU)/da = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    da = dc * b * (sig_a * (1.0 + a * (1.0 - sig_a)))
    
    # Store gradients
    tl.store(da_ptr + col_offsets, da, mask=mask)
    tl.store(db_ptr + col_offsets, db, mask=mask)

def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of SWiGLU operation.
    Args:
        a: First input tensor
        b: Second input tensor
    Returns:
        Output tensor c = SiLU(a) * b
    """
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    
    # Handle input reshaping
    orig_shape = a.shape
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    
    # Prepare output tensor
    c = torch.empty_like(a)
    
    # Calculate grid and block sizes
    n_rows, n_cols = a.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    
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

def swiglu_backward(grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """
    Backward pass of SWiGLU operation.
    Args:
        grad_output: Gradient of the loss with respect to the output
        a: First input tensor from forward pass
        b: Second input tensor from forward pass
    Returns:
        Tuple of gradients with respect to inputs (da, db)
    """
    assert grad_output.is_cuda and a.is_cuda and b.is_cuda
    
    # Handle input reshaping
    orig_shape = grad_output.shape
    grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    
    # Prepare gradient tensors
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    
    # Calculate grid and block sizes
    n_rows, n_cols = grad_output.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    
    # Launch kernel
    _swiglu_backward_kernel[(n_rows,)](
        grad_output.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        da.data_ptr(),
        db.data_ptr(),
        grad_output.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return da.reshape(orig_shape), db.reshape(orig_shape)
