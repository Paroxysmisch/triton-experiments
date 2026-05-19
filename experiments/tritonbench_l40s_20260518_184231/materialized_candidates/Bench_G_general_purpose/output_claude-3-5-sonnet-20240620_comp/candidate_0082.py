import torch
import triton
import triton.language as tl

@triton.jit
def load_reduce_kernel(
    x_ptr,  # pointer to input matrix
    y_ptr,  # pointer to output vector
    stride_xm,  # stride for rows
    stride_xn,  # stride for columns
    stride_y,   # stride for output
    M,  # number of rows
    N,  # number of columns
    BLOCK_M: tl.constexpr,  # block size for rows
    BLOCK_N: tl.constexpr,  # block size for columns
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate row index
    row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Create mask for valid rows
    row_mask = row_idx < M
    
    # Initialize output with minimum value
    output = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    
    # Iterate over columns in blocks
    for n in range(0, N, BLOCK_N):
        # Create column indices
        col_idx = n + tl.arange(0, BLOCK_N)
        # Create mask for valid columns
        col_mask = col_idx < N
        
        # Combined mask for valid elements
        mask = row_mask[:, None] & col_mask[None, :]
        
        # Calculate offsets for current block
        offs = row_idx[:, None] * stride_xm + col_idx[None, :] * stride_xn
        
        # Load block of data
        x = tl.load(x_ptr + offs, mask=mask, other=float("-inf"))
        
        # Update maximum values
        output = tl.maximum(output, tl.max(x, axis=1))
    
    # Store results for valid rows
    tl.store(y_ptr + row_idx * stride_y, output, mask=row_mask)

def load_reduce(x):
    """
    Compute row-wise maximum of input tensor x using Triton kernel
    
    Args:
        x: Input tensor of shape (M, N)
    Returns:
        y: Output tensor of shape (M,) containing row-wise maxima
    """
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    
    # Define block sizes
    BLOCK_M = 32
    BLOCK_N = 128
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_M),)
    
    # Launch kernel
    load_reduce_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        stride_xm=x.stride(0),
        stride_xn=x.stride(1),
        stride_y=1,
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return y

# Test the implementation
def test_load_reduce():
    torch.manual_seed(0)
    M, N = 1024, 2048
    x = torch.randn(M, N, device='cuda')
    y_triton = load_reduce(x)
    y_torch = torch.max(x, dim=1)[0]
    torch.testing.assert_close(y_triton, y_torch)
    print("Test passed!")
