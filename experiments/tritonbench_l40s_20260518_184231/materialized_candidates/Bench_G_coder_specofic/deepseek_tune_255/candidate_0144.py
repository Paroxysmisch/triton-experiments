import torch
import triton
import triton.language as tl

@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt, dA_cumsum, dt_out,
    A,
    chunk_size,
    dt_bias,
    dt_softplus: tl.constexpr,
    dt_limit: tl.constexpr,
    batch, seqlen, nheads, nchunks,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_dt_bias_head,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_size,
    stride_dt_out_batch, stride_dt_out_seqlen, stride_dt_out_head, stride_dt_out_size,
    DT_INF: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_seqlen
    dA_cumsum += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * chunk_size * stride_dt_seqlen)
    dt_out_ptrs = dt_out + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_size)
    dA_cs_ptrs = dA_cumsum + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_size)
    chunks = tl.cdiv(seqlen, chunk_size)

    mask_h = offs_h < nheads
    mask_c = offs_c < chunks
    mask_hc = mask_h[:, None] & mask_c[None, :]

    dt = tl.load(dt_ptrs, mask=mask_hc, other=0.0).to(tl.float32)
    if dt_bias is not None:
        dt_bias = dt_bias + offs_h * stride_dt_bias_head
        bias = tl.load(dt_bias, mask=mask_h, other=0.0).to(tl.float32)
        dt += bias[:, None]
    if dt_softplus:
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
    dt = tl.minimum(tl.maximum(dt, dt_limit[0]), dt_limit[1])
    dt = tl.where(tl.isnan(dt), 0.0, dt)
    dt = tl.where(tl.isinf(dt), dt_limit[1] * torch.tensor(1.0), dt)
    dt = tl.where(dt>=DT_INF, dt_limit[1]*torch.tensor(1.0), dt)
    tl.store(dt_out_ptrs, dt, mask=mask_hc)

    A = A + offs_h * stride_dt_head
    A = tl.load(A, mask=mask_h, other=0.0).to(tl.float32)
    cumsum = tl.cumsum(dt, axis=1)
    cumsum -= tl.minimum(tl.maximum(cumsum, dt_limit[0]), dt_limit[1])
    cumsum = tl.where(cumsum>=DT_INF, dt_limit[1]*torch.tensor(1.0), cumsum)
    tl.store(dA_cs_ptrs, cumsum, mask=mask_hc)


def _chunk_cumsum_fwd(
    dt: torch.Tensor, A: torch.Tensor,
    chunk_size: int, dt_bias: torch.Tensor = None,
    dt_softplus: bool = False, dt_limit = (-10.0, 10.0)
):
    batch, seqlen, nheads = dt.shape
    nchunks = int(np.ceil(seqlen / chunk_size))
    dA_cumsum = dt.new_empty(batch, nheads, nchunks, chunk_size)
    dt_out = dt.new_empty(dt.shape)

    assert dt.stride(0) == 1, "Batch dim must be 1Byte aligned"
    assert A.stride(0) == 1, "Head dim must be 1Byte aligned"
    assert dt.is_contiguous(), "Input tensor must be contiguous"
    assert dt_out.is_contiguous(), "Output tensor must be contiguous"
    assert dA_cumsum.is_contiguous(), "dA_cumsum tensor must be contiguous"
    if dt_bias is not None:
        assert dt_bias.stride(0) == 1, "Head dim must be 1Byte aligned"
        assert dt_bias.is_contiguous(), "dt_bias tensor must be contiguous"
        assert dt_bias.shape[0] == nheads, "Bias shape must equal to head number"

    if dt_bias is None:
        dt_bias = torch.empty(0)
    
    DT_INF = np.finfo(np.float32).max

    kwargs = [dt, dA_cumsum, dt_out, A, chunk_size, dt_bias, dt_softplus, dt_limit,
              batch, seqlen, nheads, nchunks,
              dt.stride(0), dt.stride(1), dt.stride(2),
              dt_bias.stride(0),
              dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
              dt_out.stride(0), dt_out.stride(1), dt_out.stride(2), dt_out.stride(3),
              DT_INF]
    kwargs.extend([32, min(32, chunk_size)])
    grid = lambda META: (batch, META["chunk_number"], nheads)
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid](*kwargs)
    return dA_cumsum, dt_out
