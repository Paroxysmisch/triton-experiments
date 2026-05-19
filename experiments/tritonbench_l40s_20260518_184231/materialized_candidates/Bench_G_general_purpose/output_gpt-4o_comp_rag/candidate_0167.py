import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X,  # pointer to the input
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    stride_x_row,  # stride to access each row in the input
    N,  # number of elements in each row
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr  # number of elements processed per block
):
    row = tl.program_id(0)
    X += row * stride_x_row
    DY += row * stride_x_row
    DX += row * stride_x_row
    
    _var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
        tl.store(DX + cols, dx, mask=mask)

def _l2_norm_bwd(x, dy, eps):
    x = x.contiguous()
    dy = dy.contiguous()
    dx = torch.empty_like(x)
    
    M, N = x.shape
    stride_x_row = x.stride(0)
    
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This L2 norm backward pass doesn't support feature dim >= 64KB.")
    
    _l2_norm_bwd_kernel[(M,)](
        x,
        dy,
        dx,
        stride_x_row,
        N,
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )
    
    return dx
