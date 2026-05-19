import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X_ptr, Y_ptr,
    stride_x_row,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    # Get the row index
    row = tl.program_id(0)
    
    # Compute pointer offsets for this row
    row_start_ptr = X_ptr + row * stride_x_row
    
    # Initialize accumulator for sum of squares
    sum_squares = tl.zeros([1], dtype=tl.float32)
    
    # Load and process data in blocks
    for block_idx in range(0, N, BLOCK_N):
        # Create block mask
        block_mask = block_idx + tl.arange(0, BLOCK_N) < N
        
        # Load input values
        x = tl.load(row_start_ptr + block_idx, mask=block_mask, other=0.0)
        
        # Accumulate sum of squares
        sum_squares += tl.sum(x * x, axis=0)
    
    # Compute L2 norm (add epsilon for numerical stability)
    norm = tl.sqrt(sum_squares + eps)
    
    # Normalize and store results
    for block_idx in range(0, N, BLOCK_N):
        block_mask = block_idx + tl.arange(0, BLOCK_N) < N
        
        # Load input values again
        x = tl.load(row_start_ptr + block_idx, mask=block_mask, other=0.0)
        
        # Normalize
        y = x / norm
        
        # Store results
        tl.store(Y_ptr + row * stride_x_row + block_idx, y, mask=block_mask)

def _l2_norm_fwd(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute L2 normalization of input tensor x along the last dimension.
    
    Args:
        x: Input tensor of shape (..., N)
        eps: Small epsilon value for numerical stability
    
    Returns:
        y: L2 normalized tensor of the same shape as input
    """
    # Ensure input is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Get input dimensions
    *batch_dims, N = x.shape
    M = 1
    for dim in batch_dims:
        M *= dim
    
    # Prepare output tensor
    y = torch.empty_like(x)
    
    # Compute maximum block size (power of 2)
    BLOCK_N = triton.next_power_of_2(min(N, 1024))
    
    # Launch kernel
    _l2_norm_fwd_1pass_kernel[(M,)](
        x, y,
        x.stride(-2) if len(x.shape) > 1 else 0,
        N, eps,
        BLOCK_N=BLOCK_N,
    )
    
    return y

# Example usage
if __name__ == "__main__":
    # Create sample input tensor
    x = torch.randn(2, 3, 512, device='cuda')
    
    # Apply L2 normalization
    y = _l2_norm_fwd(x)
    
    # Verify results
    y_ref = torch.nn.functional.normalize(x, p=2, dim=-1)
    assert torch.allclose(y, y_ref, rtol=1e-3, atol=1e-3)
