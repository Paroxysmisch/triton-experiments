import torch
import triton
import triton.language as tl

@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt_ptr, A_ptr, dt_bias_ptr, dA_cumsum_ptr, dt_out_ptr,
    DT_SOFTPLUS: tl.constexpr, BIAS_ENABLED: tl.constexpr,
    DT_MIN, DT_MAX,
    BATCH: tl.constexpr, NHEADS: tl.constexpr, NCHUNKS: tl.constexpr, CHUNK_SIZE: tl.constexpr, SEQLEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_ch = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NHEADS
    h = pid_bh % NHEADS
    x = tl.arange(0, BLOCK_SIZE)
    offset_seq = pid_ch * CHUNK_SIZE + x
    offset_dt = b * SEQLEN * NHEADS + offset_seq * NHEADS + h
    mask = offset_seq < SEQLEN

    bias_val = 0.0
    if BIAS_ENABLED:
        bias_val = tl.load(dt_bias_ptr + h)
    scale_val = tl.load(A_ptr + h)

    dt_val = tl.where(mask, tl.load(dt_ptr + offset_dt, mask=mask), 0.0)
    if BIAS_ENABLED:
        dt_val = dt_val + bias_val
    if DT_SOFTPLUS:
        dt_val = tl.log(1 + tl.exp(dt_val))
    dt_val = tl.maximum(dt_val, DT_MIN)
    dt_val = tl.minimum(dt_val, DT_MAX)
    dt_scaled = dt_val * scale_val

    partial_sum = 0.0
    for i in range(BLOCK_SIZE):
        valid_i = offset_seq[i] < SEQLEN
        val_i = dt_scaled[i] if valid_i else 0.0
        partial_sum += val_i
        out_offset = b * NHEADS * NCHUNKS * CHUNK_SIZE + h * NCHUNKS * CHUNK_SIZE + pid_ch * CHUNK_SIZE + i
        if valid_i:
            tl.store(dA_cumsum_ptr + out_offset, partial_sum)
            tl.store(dt_out_ptr + out_offset, dt_val[i])

def _chunk_cumsum_fwd(
    dt: torch.Tensor,
    A: torch.Tensor,
    chunk_size: int,
    dt_bias: torch.Tensor = None,
    dt_softplus: bool = False,
    dt_limit: tuple = (-1e30, 1e30)
):
    b, seqlen, nheads = dt.shape
    n_chunks = (seqlen + chunk_size - 1) // chunk_size
    dA_cumsum = torch.empty((b, nheads, n_chunks, chunk_size), dtype=dt.dtype, device=dt.device)
    dt_out = torch.empty((b, nheads, n_chunks, chunk_size), dtype=dt.dtype, device=dt.device)
    bias_enabled = dt_bias is not None
    if dt_bias is None:
        dt_bias = dt.new_zeros((nheads,))
    grid = (n_chunks, b * nheads)
    block_size = chunk_size
    _chunk_cumsum_fwd_kernel[grid](
        dt, A, dt_bias, dA_cumsum, dt_out,
        dt_softplus, bias_enabled,
        dt_limit[0], dt_limit[1],
        b, nheads, n_chunks, chunk_size, seqlen,
        BLOCK_SIZE=block_size
    )
    return dA_cumsum, dt_out
