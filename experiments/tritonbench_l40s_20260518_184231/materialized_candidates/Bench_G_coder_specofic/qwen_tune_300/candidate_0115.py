import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr,
    rms_w_ptr,
    out_ptr,
    N_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M_SIZE: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M_SIZE + tl.arange(0, BLOCK_M_SIZE)
    offs_n = pid_n * BLOCK_N_SIZE + tl.arange(0, BLOCK_N_SIZE)
    rms_w_ptrs = rms_w_ptr + offs_n
    rms_w = tl.load(rms_w_ptrs, mask=offs_n < N_SIZE, other=1.0)
    x_ptrs = x_ptr + offs_m[:, None] * N_SIZE + offs_n[None, :]
    x = tl.load(x_ptrs, mask=offs_n[None, :] < N_SIZE, other=0.0)
    x = tl.reshape(x, (BLOCK_M_SIZE, BLOCK_N_SIZE))
    x_zm = tl.where(offs_n[None, :] < N_SIZE, x, 0.0)
    var = tl.sum(x_zm * x_zm, axis=1) / N_SIZE
    rstd = 1 / tl.sqrt(var + eps)
    x_hat = x_zm * rstd
    out = x_hat * rms_w
    out_ptrs = out_ptr + offs_m[:, None] * N_SIZE + offs_n[None, :]
    tl.store(out_ptrs, out, mask=offs_n[None, :] < N_SIZE)

def rmsnorm_wrapper(x, rms_weights, eps=1e-6):
    out = torch.empty_like(x)
    x_strides = x.stride()
    M, N = x_strides[-2], x_strides[-1]
    x = x.reshape(-1, M, N)
    batch, M, N = x.shape
    grid = (batch, M)
    rms_weights = rms_weights.reshape(-1, 1, rms_weights.shape[-1])
    rms_weights = rms_weights.expand(batch, M, -1)
    rmsnorm_triton[grid](
        x,
        rms_weights,
        out,
        eps=eps,
        N_SIZE=N,
        BLOCK_M_SIZE=8,
        BLOCK_N_SIZE=32,
        num_warps=4,
    )
    return out
