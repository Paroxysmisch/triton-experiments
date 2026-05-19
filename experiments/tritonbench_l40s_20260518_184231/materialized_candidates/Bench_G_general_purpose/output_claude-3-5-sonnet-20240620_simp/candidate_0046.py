import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X_ptr,  # Pointer to input tensor [M, N]
    Y_ptr,  # Pointer to output tensor [M, N]
    M,      # Number of rows
    N,      # Number of columns
    stride_xm, stride_xn,  # Strides for input tensor
    stride_ym, stride_yn,  # Strides for output tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process in parallel
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute row index
    row = pid
    
    # Don't process if row is out of bounds
    if row >= M:
        return
        
    # Compute pointers for current row
    x_row_ptr = X_ptr + row * stride_xm
    y_row_ptr = Y_ptr + row * stride_ym
    
    # Initialize accumulator for squared sum
    acc = 0.0
    
    # First pass: compute squared sum
    for idx in range(0, N, BLOCK_SIZE):
        # Create block mask
        mask = idx + tl.arange(0, BLOCK_SIZE) < N
        
        # Load input values
        x = tl.load(x_row_ptr + idx * stride_xn, mask=mask, other=0.0)
        
        # Accumulate squared values
        acc += tl.sum(x * x, mask=mask)
    
    # Compute L2 norm (sqrt of squared sum)
    norm = tl.sqrt(acc + 1e-12)  # Add epsilon for numerical stability
    
    # Second pass: normalize and store results
    for idx in range(0, N, BLOCK_SIZE):
        mask = idx + tl.arange(0, BLOCK_SIZE) < N
        x = tl.load(x_row_ptr + idx * stride_xn, mask=mask, other=0.0)
        
        # Normalize and store
        y = x / norm
        tl.store(y_row_ptr + idx * stride_yn, y, mask=mask)

@triton.jit
def _l2_norm_bwd_kernel(
    X_ptr,   # Pointer to input tensor [M, N]
    DY_ptr,  # Pointer to gradient tensor [M, N]
    DX_ptr,  # Pointer to output gradient tensor [M, N]
    M,       # Number of rows
    N,       # Number of columns
    stride_xm, stride_xn,   # Strides for input tensor
    stride_dym, stride_dyn, # Strides for gradient tensor
    stride_dxm, stride_dxn, # Strides for output gradient tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    row = pid
    
    if row >= M:
        return
        
    # Compute row pointers
    x_row_ptr = X_ptr + row * stride_xm
    dy_row_ptr = DY_ptr + row * stride_dym
    dx_row_ptr = DX_ptr + row * stride_dxm
    
    # First pass: compute squared sum and dot product
    squared_sum = 0.0
    dot_product = 0.0
    
    for idx in range(0, N, BLOCK_SIZE):
        mask = idx + tl.arange(0, BLOCK_SIZE) < N
        
        x = tl.load(x_row_ptr + idx * stride_xn, mask=mask, other=0.0)
        dy = tl.load(dy_row_ptr + idx * stride_dyn, mask=mask, other=0.0)
        
        squared_sum += tl.sum(x * x, mask=mask)
        dot_product += tl.sum(x * dy, mask=mask)
    
    # Compute normalization factor
    norm = tl.sqrt(squared_sum + 1e-12)
    inv_norm = 1.0 / norm
    
    # Second pass: compute and store gradients
    for idx in range(0, N, BLOCK_SIZE):
        mask = idx + tl.arange(0, BLOCK_SIZE) < N
        
        x = tl.load(x_row_ptr + idx * stride_xn, mask=mask, other=0.0)
        dy = tl.load(dy_row_ptr + idx * stride_dyn, mask=mask, other=0.0)
        
        # Compute gradient according to chain rule
        dx = (dy * inv_norm - x * inv_norm * inv_norm * dot_product)
        
        # Store result
        tl.store(dx_row_ptr + idx * stride_dxn, dx, mask=mask)

# Wrapper functions
def _l2_norm_fwd(x: torch.Tensor) -> torch.Tensor:
    # Get input dimensions
    M, N = x.shape
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Launch kernel
    BLOCK_SIZE = 128
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    # Get input dimensions
    M, N = x.shape
    
    # Allocate output gradient
    dx = torch.empty_like(x)
    
    # Launch kernel
    BLOCK_SIZE = 128
    grid = (M,)
    _l2_norm_bwd_kernel[grid](
        x, dy, dx,
        M, N,
        x.stride(0), x.stride(1),
        dy.stride(0), dy.stride(1),
        dx.stride(0), dx.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dx
