import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX,
    stride_x_row, stride_dy_row, stride_dx_row,
    M, N, eps,
    BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return
    
    # Compute pointers to current row
    x_row = X + row_idx * stride_x_row
    dy_row = DY + row_idx * stride_dy_row
    dx_row = DX + row_idx * stride_dx_row
    
    # Load data with mask handling
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    x = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_row + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute intermediate values
    sum_x_sq = tl.sum(x * x)
    sum_dy_x = tl.sum(dy * x)
    var = sum_x_sq
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Calculate final gradients
    inv_var_eps = 1.0 / (var + eps)
    dx = dy * rstd - (sum_dy_x * inv_var_eps * rstd) * x
    
    # Write back results with mask
    tl.store(dx_row + offsets, dx, mask=mask)

def _l2_norm_bwd(dy: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    # Reshape inputs to 2D (M, N)
    shape = x.shape
    x = x.contiguous()
    dy = dy.contiguous()
    x_2d = x.view(-1, shape[-1])
    dy_2d = dy.view(-1, shape[-1])
    M, N = x_2d.shape
    
    # Determine block size and validate
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > 4096:
        raise ValueError(f"Feature dimension too large ({N}), maximum supported is 4096 after rounding")
    
    # Allocate output tensor
    dx = torch.empty_like(x)
    dx_2d = dx.view(M, N)
    
    # Launch kernel
    grid = (M,)
    _l2_norm_bwd_kernel[grid](
        x_2d, dy_2d, dx_2d,
        x_2d.stride(0), dy_2d.stride(0), dx_2d.stride(0),
        M, N, eps,
        BLOCK_N=BLOCK_N
    )
    
    return dx.view(*shape)
