import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt_ptr, A_ptr, dt_out_ptr, dA_cumsum_ptr,
    batch, seqlen, nheads, chunk_size,
    dt_bias_ptr, dt_softplus: tl.constexpr, dt_limit_min, dt_limit_max,
    HEADS_PER_CTAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)
    dt_ptr += pid_b * seqlen * nheads + pid_c * chunk_size * nheads
    dt_out_ptr += pid_b * seqlen * nheads + pid_c * chunk_size * nheads
    dA_cumsum_ptr += pid_b * nheads * chunk_size + pid_c * chunk_size * nheads

    A_cumsum_ptr = dA_cumsum_ptr
    A_ptr += pid_c * chunk_size * nheads
    A_cumsum_ptr += pid_c * chunk_size * nheads

    chunk_size_limit = min(chunk_size, seqlen - pid_m * chunk_size)

    head_offset = tl.arange(0, HEADS_PER_CTAS)
    dt_ptr += head_offset
    dt_out_ptr += head_offset
    A_cumsum_ptr += head_offset
    A_ptr += head_offset

    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    dt_ptrs = dt_ptr + (offs_m[:, None] * nheads + offs_n[None, :]) % nheads
    dt_ptrs = dt_ptrs + (pid_m * BLOCK_SIZE_M * nheads)
    dt_out_ptrs = dt_out_ptr + (offs_m[:, None] * nheads + offs_n[None, :]) % nheads
    dt_out_ptrs = dt_out_ptrs + (pid_m * BLOCK_SIZE_M * nheads)
    A_ptrs = A_ptr + offs_n
    A_cumsum_ptrs = A_cumsum_ptr + offs_n
    dt_bias_ptrs = dt_bias_ptr + offs_n

    dt = tl.load(dt_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < HEADS_PER_CTAS), other=0.0)
    if dt_softplus:
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
    if dt_bias_ptr is not None:
        dt_bias = tl.load(dt_bias_ptrs, mask=(offs_n < HEADS_PER_CTAS), other=0.0)
        dt += dt_bias[None, :]
    dt = tl.minimum(tl.maximum(dt, dt_limit_min), dt_limit_max)
    if pid_m == 0:
        dt_out = tl.zeros((BLOCK_SIZE_M, HEADS_PER_CTAS), dtype=tl.float32)
    else:
        dt_out = tl.load(dt_out_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < HEADS_PER_CTAS),
                         other=0.0)
    dt_out += dt
    tl.store(dt_out_ptrs, dt_out, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < HEADS_PER_CTAS))
    A = tl.load(A_ptrs, mask=(offs_n < HEADS_PER_CTAS), other=0.0)
    A = A[:, None]
    A_cumsum = tl.cumsum(A, axis=0)
    tl.store(A_cumsum_ptrs, A_cumsum, mask=(offs_n < HEADS_PER_CTAS))


def _chunk_cumsum_fwd(dt: Tensor, A: Tensor, chunk_size: int, dt_bias: Tensor = None, dt_softplus: bool = False,
                      dt_limit: Tuple[float, float] = (0.0, 1.0)) -> Tuple[Tensor, Tensor]:
    B, S, H = dt.shape
    assert H % 32 == 0
    assert chunk_size > 0 and S % chunk_size == 0
    nheads = H
    nchunks = S // chunk_size
    dA_cumsum = torch.empty(B, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dt_out = torch.empty(B, nchunks, chunk_size, nheads, device=dt.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), B, nchunks)
    HEADS_PER_CTAS = 32 if nheads <= 64 else 16
    BLOCK_SIZE_M = 128 if nheads <= 64 else 64
    _chunk_cumsum_fwd_kernel[grid](
        dt, A, dt_out, dA_cumsum,
        B, S, nheads, chunk_size,
        dt_bias, dt_softplus, dt_limit[0], dt_limit[1],
        HEADS_PER_CTAS=HEADS_PER_CTAS,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=32 if nheads <= 64 else 16,
    )
    dt_out = torch.reshape(dt_out, (B, S, nheads))
    return dA_cumsum, dt_out
