import torch
import triton
import triton.language as tl

def cfggen():
    block_ns = [16, 32, 64, 128, 256, 512]
    warps = [1, 2, 4, 8, 16]
    configs = [
        triton.Config({"BLOCK_N": bn}, num_warps=w)
        for bn in block_ns
        for w in warps
    ]
    return configs

@triton.autotune(configs=cfggen(), key=["N"])
@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, stride_x_row, stride_dx_row, stride_dy_row, stride_dx_row, M, N, eps=1e-5
):
    # set offset
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offset = pid * stride_x_row + offs_n
    x = tl.load(X + offset, mask=offs_n < N, other=0.0).to(tl.float32)
    dy = tl.load(DY + offset, mask=offs_n < N, other=0.0).to(tl.float32)
    # compute variance
    x_var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)
    # compute dx
    dx = dy * rstd - tl.sum(dy * x, axis=0) * (1.0 / (x_var + eps)) * rstd * x
    # write-back dx
    tl.store(DX + offset, dx, mask=offs_n < N)

def _l2_norm_bwd(x, dy):
    # make sure input tensor is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not dy.is_contiguous():
        dy = dy.contiguous()
    # check if feature dim is larger than block size
    M, N = x.shape
    if N > BLOCK_N:
        raise ValueError(f"feature dim {N} > BLOCK_N {BLOCK_N}")
    # reshape input data into 2D tensor
    x = x.view(-1, N)
    dy = dy.view(-1, N)
    dx = torch.empty_like(x)
    # launch kernel
    grid = (triton.cdiv(M, BLOCK_M), 1)
    _l2_norm_bwd_kernel[grid](
        x,
        dy,
        dx,
        x.stride(0),
        dx.stride(0),
        dy.stride(0),
        dx.stride(0),
        M,
        N,
    )
    # reshape back to original shape
    dx = dx.view(x.shape)
    return dx
