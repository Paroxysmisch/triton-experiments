import triton
import triton.language as tl
import torch
import math
from typing import Tuple

@triton.jit
def _rmsnorm_fwd_kernel(
    X, Y, W, Rstd,           # pointers to input, output, weights, and 1/std
    stride_x_row,            # stride for moving between rows in X
    stride_y_row,            # stride for moving between rows in Y
    N,                       # number of columns
    eps,                     # epsilon for numerical stability
    BLOCK_N: tl.constexpr,   # block size (compile-time constant)
    IS_EVEN_N: tl.constexpr  # whether N is evenly divisible by BLOCK_N
):
    # Get the row index for this program instance
    row = tl.program_id(0)
    
    # Offset input/output pointers to the correct row
    X += row * stride_x_row
    Y += row * stride_y_row

    # Create a range for accessing columns
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    # Load input values
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute variance
    x2 = tl.where(mask, x * x, 0.0)
    var = tl.sum(x2, axis=0) / N
    
    # Compute reciprocal standard deviation
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    
    # Load weights and normalize
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    y = x * rstd * w
    
    # Store output
    tl.store(Y + cols, y, mask=mask)

@triton.jit
def _rmsnorm_bwd_kernel(
    X, W, DY, DX, DW, Rstd,  # pointers to tensors
    stride_x_row,            # strides for moving between rows
    stride_dy_row,
    stride_dx_row,
    M, N,                    # dimensions
    eps,                     # epsilon
    rows_per_program,        # rows to process per program instance
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr
):
    # Get program index and calculate row range
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, M)
    
    # Setup column indexing
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    # Initialize pointers
    X += row_start * stride_x_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    
    # Load weights
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    
    # Initialize weight gradients
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # Process each row in the block
    for row in range(row_start, row_end):
        # Load input and gradient
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        rstd = tl.load(Rstd + row)
        
        # Compute normalized input
        xhat = x * rstd
        
        # Compute gradients
        wdy = w * dy
        dw += dy * xhat
        
        # Compute input gradient
        dx_bar = wdy * rstd
        dx_hat_sum = tl.sum(xhat * wdy, axis=0) / N
        dx = dx_bar - xhat * dx_hat_sum
        
        # Store input gradient
        tl.store(DX + cols, dx, mask=mask)
        
        # Update pointers
        X += stride_x_row
        DY += stride_dy_row
        DX += stride_dx_row
    
    # Store weight gradients
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)

def rmsnorm_triton_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    
    # Allocate output tensors
    y = torch.empty_like(x)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    
    # Calculate block size
    BLOCK_N = triton.next_power_of_2(min(N, 65536 // x.element_size()))
    
    # Launch kernel
    grid = (M,)
    _rmsnorm_fwd_kernel[grid](
        x, y, weight, rstd,
        x.stride(0), y.stride(0),
        N, eps, BLOCK_N, (N % BLOCK_N == 0)
    )
    
    return y, rstd

def rmsnorm_triton_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    
    # Allocate output tensors
    dx = torch.empty_like(x)
    
    # Calculate number of SM blocks and rows per program
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    dw_partial = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
    rows_per_program = math.ceil(M / sm_count)
    
    # Calculate block size
    BLOCK_N = triton.next_power_of_2(min(N, 65536 // x.element_size()))
    
    # Launch kernel
    grid = (sm_count,)
    _rmsnorm_bwd_kernel[grid](
        x, weight, dy, dx, dw_partial, rstd,
        x.stride(0), dy.stride(0), dx.stride(0),
        M, N, eps, rows_per_program, BLOCK_N, (N % BLOCK_N == 0)
    )
    
    # Reduce partial weight gradients
    dw = dw_partial.sum(0).to(weight.dtype)
    
    return dx, dw
