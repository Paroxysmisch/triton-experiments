import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(X_ptr, Y_ptr, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    # Compute the row index for the current program
    row_idx = tl.program_id(0)
    
    # Compute the offset for the current row in X and Y
    x_row_ptr = X_ptr + row_idx * stride_x_row
    y_row_ptr = Y_ptr + row_idx * stride_x_row
    
    # Initialize sum of squares
    sum_squares = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # Iterate over the columns in blocks of size BLOCK_N
    for col_start in range(0, N, BLOCK_N):
        # Load a block of data from X
        x_block = tl.load(x_row_ptr + col_start, mask=col_start + tl.arange(0, BLOCK_N) < N, other=0.0)
        
        # Update sum of squares
        sum_squares += x_block * x_block
    
    # Compute variance and rstd
    variance = tl.sum(sum_squares)
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize the block of data and store in Y
    for col_start in range(0, N, BLOCK_N):
        x_block = tl.load(x_row_ptr + col_start, mask=col_start + tl.arange(0, BLOCK_N) < N, other=0.0)
        y_block = x_block * rstd
        tl.store(y_row_ptr + col_start, y_block, mask=col_start + tl.arange(0, BLOCK_N) < N)

import torch

def _l2_norm_fwd(x, eps=1e-5):
    # Ensure x is a 2D tensor and make it contiguous
    x = x.view(-1, x.size(-1)).contiguous()
    M, N = x.shape
    
    # Calculate BLOCK_N
    element_size = x.element_size()
    max_block_size = 64 * 1024  # 64KB
    BLOCK_N = max_block_size // element_size
    if BLOCK_N > N:
        BLOCK_N = N
    if N > BLOCK_N:
        raise ValueError("Feature dimension N exceeds BLOCK_N limit.")
    
    # Initialize output tensor
    y = torch.empty_like(x)
    
    # Launch the kernel
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x, y, x.stride(0), N, eps, BLOCK_N=BLOCK_N
    )
    
    # Return the normalized tensor reshaped to original dimensions
    return y.view_as(x)

# Example usage
x = torch.randn(128, 256, device='cuda', dtype=torch.float32)
y = _l2_norm_fwd(x)
