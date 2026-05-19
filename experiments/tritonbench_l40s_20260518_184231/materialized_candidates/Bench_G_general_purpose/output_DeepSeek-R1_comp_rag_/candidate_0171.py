import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X,  # pointer to input tensor
    DY,  # pointer to gradient of output
    DX,  # pointer to gradient of input (to be computed)
    stride_x_row,  # stride between rows of X
    N,  # number of elements per row
    eps,  # epsilon for numerical stability
    BLOCK_N: tl.constexpr,  # block size for processing elements
):
    row = tl.program_id(0)
    row_start = row * stride_x_row
    X_row = X + row_start
    DY_row = DY + row_start
    DX_row = DX + row_start

    # Accumulate sum of x squared and sum of dy * x
    sum_x2 = 0.0
    sum_dy_x = 0.0
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_x2 += tl.sum(x * x, axis=0)
        sum_dy_x += tl.sum(dy * x, axis=0)
    
    var = sum_x2
    rstd = 1.0 / tl.sqrt(var + eps)
    factor = sum_dy_x * (1.0 / (var + eps)) * rstd

    # Compute dx for each element in the row
    for offset in range(0, N, BLOCK_N):
        cols = offset + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_row + cols, mask=mask, other=0.0).to(tl.float32)
        dx = dy * rstd - factor * x
        tl.store(DX_row + cols, dx, mask=mask)

def _l2_norm_bwd(dy: torch.Tensor, x: torch.Tensor, eps: float):
    # Ensure contiguous memory layout
    x = x.contiguous()
    dy = dy.contiguous()
    
    # Reshape input tensors to 2D
    x_2d = x.view(-1, x.size(-1))
    M, N = x_2d.shape
    
    # Determine block size
    max_fused_size = 65536 // x.element_size()
    BLOCK_N = min(max_fused_size, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension {N} exceeds maximum block size {BLOCK_N}.")
    
    # Allocate output tensor
    dx = torch.empty_like(x)
    
    # Configure kernel launch parameters
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M,)
    
    # Launch kernel
    _l2_norm_bwd_kernel[grid](
        x_2d, dy.view_as(x_2d), dx.view_as(x_2d),
        x_2d.stride(0), N, eps,
        BLOCK_N=BLOCK_N, num_warps=num_warps
    )
    
    return dx.view_as(x)

# Example usage:
# x = torch.randn(10, 512, device='cuda', dtype=torch.float16)
# dy = torch.randn_like(x)
# eps = 1e-5
# dx = _l2_norm_bwd(dy, x, eps)
