import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(X_ptr, DY_ptr, DX_ptr, N, eps, BLOCK_SIZE: tl.constexpr):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Compute the offset for the row
    offset = row_idx * N
    
    # Load a slice of X and DY for the current row
    X = tl.load(X_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)
    DY = tl.load(DY_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)
    
    # Compute the squared norm of the row
    squared_norm = tl.sum(X * X, axis=0)
    
    # Compute the variance with epsilon for numerical stability
    variance = tl.sqrt(squared_norm + eps)
    
    # Compute the gradient
    grad = (DY - tl.sum(DY * X, axis=0) * X / (variance * variance)) / variance
    
    # Store the result in DX
    tl.store(DX_ptr + offset + tl.arange(0, BLOCK_SIZE), grad, mask=tl.arange(0, BLOCK_SIZE) < N)

def _l2_norm_bwd(x, dy, eps=1e-5):
    # Get the shape of the input
    B, N = x.shape
    
    # Allocate memory for the output gradient
    dx = torch.empty_like(x)
    
    # Configure block size
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Launch the kernel
    _l2_norm_bwd_kernel[(B,)](x, dy, dx, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    
    return dx
