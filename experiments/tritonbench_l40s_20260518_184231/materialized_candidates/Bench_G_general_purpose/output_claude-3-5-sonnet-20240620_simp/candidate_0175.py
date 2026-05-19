import triton
import triton.language as tl
import torch

@triton.jit
def _l2_norm_bwd_kernel(
    X_ptr, DY_ptr, DX_ptr,
    stride_xm, stride_xn,
    stride_dym, stride_dyn,
    stride_dxm, stride_dxn,
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute row index
    row = pid
    
    # Pointers for the current row
    x_row_ptr = X_ptr + row * stride_xm
    dy_row_ptr = DY_ptr + row * stride_dym
    dx_row_ptr = DX_ptr + row * stride_dxm
    
    # Load row data
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(x_row_ptr + cols * stride_xn, mask=mask, other=0.0)
    dy = tl.load(dy_row_ptr + cols * stride_dyn, mask=mask, other=0.0)
    
    # Compute L2 norm
    x_squared = x * x
    row_sum = tl.sum(x_squared, axis=0)
    norm = tl.sqrt(row_sum + eps)
    
    # Compute gradients
    norm_inv = 1.0 / norm
    norm_inv_cube = norm_inv * norm_inv * norm_inv
    
    dot_product = tl.sum(x * dy, axis=0)
    
    # Backward formula: dx = (dy/norm - x * dot_product/norm^3)
    dx = norm_inv * dy - (x * dot_product * norm_inv_cube)
    
    # Store results
    tl.store(dx_row_ptr + cols * stride_dxn, dx, mask=mask)

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Get input dimensions
    M, N = x.shape
    
    # Allocate output
    dx = torch.empty_like(x)
    
    # Define block size (can be tuned)
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Launch kernel
    grid = (M,)
    
    _l2_norm_bwd_kernel[grid](
        x, dy, dx,
        x.stride(0), x.stride(1),
        dy.stride(0), dy.stride(1),
        dx.stride(0), dx.stride(1),
        M, N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dx

# Example usage
def l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute backward pass of L2 normalization.
    
    Args:
        x: Input tensor of shape (M, N)
        dy: Gradient tensor of shape (M, N)
        eps: Small constant for numerical stability
        
    Returns:
        dx: Gradient tensor of shape (M, N)
    """
    assert x.dim() == 2, "Input must be 2-dimensional"
    assert x.shape == dy.shape, "Input and gradient must have the same shape"
    assert x.is_cuda and dy.is_cuda, "Inputs must be CUDA tensors"
    
    return _l2_norm_bwd(x, dy, eps)
