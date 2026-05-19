import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X_ptr,  # Pointer to input tensor [B, D]
    Y_ptr,  # Pointer to output tensor [B, D]
    Var_ptr,  # Pointer to variance tensor [B]
    Rstd_ptr,  # Pointer to reciprocal std tensor [B]
    stride_xb,  # Stride for batch dimension of X
    stride_yb,  # Stride for batch dimension of Y
    D,  # Feature dimension size
    BLOCK_SIZE: tl.constexpr,  # Static block size for compilation
):
    # Get the row index
    row = tl.program_id(0)
    
    # Compute memory offsets
    x_row_off = row * stride_xb
    y_row_off = row * stride_yb
    
    # Create block-level offsets
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D
    
    # Initialize accumulator for variance
    var = tl.zeros([1], dtype=tl.float32)
    
    # Load and square elements
    for block_start in range(0, D, BLOCK_SIZE):
        x = tl.load(X_ptr + x_row_off + block_start + offs, mask=mask & (block_start + offs < D))
        var += tl.sum(x * x, axis=0)
    
    # Store variance and compute rstd
    tl.store(Var_ptr + row, var)
    rstd = tl.math.rsqrt(var + 1e-5)
    tl.store(Rstd_ptr + row, rstd)
    
    # Normalize and store output
    for block_start in range(0, D, BLOCK_SIZE):
        x = tl.load(X_ptr + x_row_off + block_start + offs, mask=mask & (block_start + offs < D))
        y = x * rstd
        tl.store(Y_ptr + y_row_off + block_start + offs, y, mask=mask & (block_start + offs < D))

def _l2_norm_fwd(x: torch.Tensor):
    # Input validation and reshaping
    if x.dim() != 2:
        x = x.reshape(x.shape[0], -1)
    
    B, D = x.shape
    # Ensure feature dimension fits in Triton's memory limits
    assert D < 65536, "Feature dimension must be less than 64KB"
    
    # Prepare output tensors
    y = torch.empty_like(x)
    var = torch.empty(B, dtype=x.dtype, device=x.device)
    rstd = torch.empty(B, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_SIZE = min(D, 1024)
    grid = (B,)
    _l2_norm_fwd_1pass_kernel[grid](
        x, y, var, rstd,
        x.stride(0), y.stride(0),
        D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y, var, rstd

@triton.jit
def _l2_norm_bwd_kernel(
    X_ptr,    # Input tensor pointer
    DY_ptr,   # Gradient tensor pointer
    DX_ptr,   # Output gradient tensor pointer
    Var_ptr,  # Variance tensor pointer
    Rstd_ptr, # Reciprocal std tensor pointer
    stride_xb,  # Stride for batch dimension of X
    stride_dyb, # Stride for batch dimension of DY
    stride_dxb, # Stride for batch dimension of DX
    D,         # Feature dimension size
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row = tl.program_id(0)
    
    # Compute memory offsets
    x_row_off = row * stride_xb
    dy_row_off = row * stride_dyb
    dx_row_off = row * stride_dxb
    
    # Load precomputed values
    var = tl.load(Var_ptr + row)
    rstd = tl.load(Rstd_ptr + row)
    
    # Create block-level offsets
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D
    
    # Initialize accumulators
    dot_product = tl.zeros([1], dtype=tl.float32)
    
    # Compute dot product of x and dy
    for block_start in range(0, D, BLOCK_SIZE):
        curr_mask = mask & (block_start + offs < D)
        x = tl.load(X_ptr + x_row_off + block_start + offs, mask=curr_mask)
        dy = tl.load(DY_ptr + dy_row_off + block_start + offs, mask=curr_mask)
        dot_product += tl.sum(x * dy, axis=0)
    
    dot_product = dot_product * rstd * rstd * rstd
    
    # Compute and store gradients
    for block_start in range(0, D, BLOCK_SIZE):
        curr_mask = mask & (block_start + offs < D)
        x = tl.load(X_ptr + x_row_off + block_start + offs, mask=curr_mask)
        dy = tl.load(DY_ptr + dy_row_off + block_start + offs, mask=curr_mask)
        
        dx = (dy - x * dot_product) * rstd
        tl.store(DX_ptr + dx_row_off + block_start + offs, dx, mask=curr_mask)

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, var: torch.Tensor, rstd: torch.Tensor):
    # Input validation and reshaping
    if x.dim() != 2:
        x = x.reshape(x.shape[0], -1)
        dy = dy.reshape(dy.shape[0], -1)
    
    B, D = x.shape
    # Ensure feature dimension fits in Triton's memory limits
    assert D < 65536, "Feature dimension must be less than 64KB"
    
    # Prepare output tensor
    dx = torch.empty_like(x)
    
    # Launch kernel
    BLOCK_SIZE = min(D, 1024)
    grid = (B,)
    _l2_norm_bwd_kernel[grid](
        x, dy, dx, var, rstd,
        x.stride(0), dy.stride(0), dx.stride(0),
        D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dx
