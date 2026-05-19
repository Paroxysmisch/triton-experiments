@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1})
        triton.Config({'BLOCK_SIZE_H': 2})
        triton.Config({'BLOCK_SIZE_H': 4})
        triton.Config({'BLOCK_SIZE_H': 8})
        triton.Config({'BLOCK_SIZE_H': 16})
        triton.Config({'BLOCK_SIZE_H': 32})
    ],
    key='chunk_size'
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt_ptr, A_ptr, dt_bias_ptr, chunks_ptr, dt_out_ptr, dA_cumsum_ptr,
    batch, seqlen, nheads, chunk_size, dt_min, dt_max, stride_dt_batch, 
    stride_dt_head, stride_A_head, stride_dt_bias_head, stride_dt_out_head, 
    stride_dA_cs_head, STRIDE_CHUNKS, DT_SOFTPLUS: tl.constexpr, HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_h = tl.program_id(axis=0)
    dt_ptr += pid_h * stride_dt_head
    dt_out_ptr += pid_h * stride_dt_out_head
    dA_cumsum_ptr += pid_h * stride_dA_cs_head
    A_ptr += pid_h * stride_A_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    chunk_starts = tl.load(chunks_ptr + pid_h * STRIDE_CHUNKS, dtype=tl.int32)
    chunk_ends = tl.load(chunks_ptr + (pid_h+1) * STRIDE_CHUNKS, dtype=tl.int32)
    for chunk in range(chunk_starts, chunk_ends):
        dt = tl.load(dt_ptr + chunk * chunk_size, dtype=tl.float32)
        if HAS_DT_BIAS:
            dt_bias = tl.load(dt_bias_ptr + chunk * chunk_size, dtype=tl.float32)
            dt += dt_bias
        if DT_SOFTPLUS:
            dt = softplus(dt)
        dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
        tl.store(dt_out_ptr + chunk * chunk_size, dt)
        A = tl.load(A_ptr)
        dA = dt * A
        dA_cs = tl.cumsum(dA)
        tl.store(dA_cumsum_ptr + chunk * chunk_size, dA_cs)

@triton.jit
def _chunk_cumsum_fwd(
    dt, A, chunks, stride_chunks, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))
):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    assert chunks.shape == (nheads+1,)
    dt_out = torch.empty_like(dt)
    dA_cumsum = torch.empty_like(dt)
    grid_chunk_cs = lambda META: (triton.cdiv(nheads, META['BLOCK_SIZE_H']),)
    _chunk_cumsum_fwd_kernel[grid_chunk_cs](
        dt, A, dt_bias, chunks, dt_out, dA_cumsum,
        batch, seqlen, nheads, chunk_size, dt_limit[0], dt_limit[1], dt.stride(0), dt.stride(2),
        A.stride(0), dt_bias.stride(0) if dt_bias is not None else 0, dt_out.stride(0), 
        dt_out.stride(2), dA_cumsum.stride(0), dA_cumsum.stride(2), STRIDE_CHUNKS, dt_softplus, 
        HAS_DT_BIAS=dt_bias is not None, BLOCK_SIZE_H=triton.next_power_of_2(chunk_size),
    )
    return dA_cumsum, dt_out
