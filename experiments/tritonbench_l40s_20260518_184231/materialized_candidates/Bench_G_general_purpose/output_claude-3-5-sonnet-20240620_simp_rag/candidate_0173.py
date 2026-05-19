import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    DX,  # pointer to gradient output
    DY,  # pointer to gradient input 
    X,   # pointer to input
    stride,  # stride between rows
    N,    # number of columns
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # block size for parallel processing
):
    # Get the row index
    row = tl.program_id(0)
    
    # Offset pointers to the current row
    X_row_ptr = X + row * stride
    DY_row_ptr = DY + row * stride
    DX_row_ptr = DX + row * stride
    
    # Compute row norm (variance)
    var = tl.zeros([1], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum(x * x, axis=0)
    
    # Add epsilon and take sqrt for stability
    norm = tl.sqrt(var + eps)
    inv_norm = 1.0 / norm
    
    # Compute dot product of X and DY
    dot = tl.zeros([1], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(x * dy, axis=0)
    
    # Compute and store gradients
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Gradient formula: (dy/norm) - (x/(norm^3)) * dot(x, dy)
        dx = (dy * inv_norm) - (x * inv_norm * inv_norm * inv_norm * dot)
        tl.store(DX_row_ptr + cols, dx, mask=mask)

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Ensure inputs are contiguous
    x = x.contiguous()
    dy = dy.contiguous()
    
    # Get input dimensions
    batch_size = x.shape[0]
    n_cols = x.shape[1]
    
    # Allocate output tensor
    dx = torch.empty_like(x)
    
    # Calculate optimal block size
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)
    
    # Launch kernel
    grid = (batch_size,)
    _l2_norm_bwd_kernel[grid](
        dx, dy, x,
        stride=n_cols,
        N=n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    
    return dx
