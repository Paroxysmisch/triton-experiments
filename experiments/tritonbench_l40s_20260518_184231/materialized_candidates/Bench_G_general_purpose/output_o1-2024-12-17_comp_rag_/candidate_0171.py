import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X,     # pointer to input
    DY,    # pointer to gradient wrt output
    DX,    # pointer to gradient wrt input
    stride_x_row,  # how to move from one row to the next
    N,     # number of columns
    eps,   # small constant for numerical stability
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    X += pid * stride_x_row
    DY += pid * stride_x_row
    DX += pid * stride_x_row

    # First pass: compute partial sums for variance and x*dy
    offs = tl.arange(0, BLOCK_N)
    partial_x2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    partial_xtdy = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        idx = start + offs
        mask = idx < N
        x = tl.load(X + idx, mask=mask, other=0.).to(tl.float32)
        dy = tl.load(DY + idx, mask=mask, other=0.).to(tl.float32)
        partial_x2 += x * x
        partial_xtdy += x * dy

    var = tl.sum(partial_x2, axis=0) / N
    sum_xtdy = tl.sum(partial_xtdy, axis=0)
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: compute dx
    for start in range(0, N, BLOCK_N):
        idx = start + offs
        mask = idx < N
        x = tl.load(X + idx, mask=mask, other=0.).to(tl.float32)
        dy = tl.load(DY + idx, mask=mask, other=0.).to(tl.float32)
        dx_val = dy * rstd - (sum_xtdy / (var + eps)) * rstd * x
        tl.store(DX + idx, dx_val, mask=mask)


def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.contiguous()
    dy = dy.contiguous()
    M, N = x.shape[:-1], x.shape[-1]
    M = int(torch.prod(torch.tensor(M))) if len(x.shape) > 1 else 1
    x_arg = x.reshape(M, N)
    dy_arg = dy.reshape(M, N)

    dx = torch.empty_like(x_arg)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This L2 norm bwd doesn't support feature dim >= 64KB.")

    num_warps = min(max(BLOCK_N // 256, 1), 8)

    _l2_norm_bwd_kernel[(M,)](
        x_arg,
        dy_arg,
        dx,
        x_arg.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return dx.reshape(x.shape)
