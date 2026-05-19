import triton
import triton.language as tl

@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt_ptr,  # pointer to the input tensor dt
    A_ptr,   # pointer to the scaling factors A
    dA_cumsum_ptr,  # pointer to the output cumulative sum result
    dt_out_ptr,  # pointer to the output modified dt
    dt_bias_ptr,  # pointer to the optional bias tensor dt_bias
    dt_softplus,  # boolean to apply softplus transformation
    dt_limit,  # clamping limits for dt
    batch,  # batch size
    seqlen,  # sequence length
    nheads,  # number of heads
    chunk_size,  # size of each chunk
    BLOCK_SIZE: tl.constexpr,  # block size for parallel processing
):
    pid = tl.program_id(axis=0)
    num_chunks = (seqlen + chunk_size - 1) // chunk_size
    chunk_id = pid % num_chunks
    head_id = (pid // num_chunks) % nheads
    batch_id = pid // (num_chunks * nheads)

    chunk_start = chunk_id * chunk_size
    chunk_end = min(chunk_start + chunk_size, seqlen)

    dt_offset = batch_id * seqlen * nheads + head_id * seqlen + chunk_start
    A_offset = head_id
    dA_cumsum_offset = batch_id * num_chunks * nheads + head_id * num_chunks + chunk_id
    dt_out_offset = batch_id * seqlen * nheads + head_id * seqlen + chunk_start

    if dt_bias_ptr is not None:
        dt_bias_offset = head_id * seqlen + chunk_start

    # Load data
    dt_chunk = tl.load(dt_ptr + dt_offset, mask=chunk_start + tl.arange(0, BLOCK_SIZE) < chunk_end, other=0.0)
    A_val = tl.load(A_ptr + A_offset)

    if dt_bias_ptr is not None:
        dt_bias_chunk = tl.load(dt_bias_ptr + dt_bias_offset, mask=chunk_start + tl.arange(0, BLOCK_SIZE) < chunk_end, other=0.0)
        dt_chunk += dt_bias_chunk

    if dt_softplus:
        dt_chunk = tl.log(1.0 + tl.exp(dt_chunk))

    dt_chunk = tl.clamp(dt_chunk, -dt_limit, dt_limit)

    # Compute cumulative sum
    dA_cumsum_chunk = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(chunk_start, chunk_end):
        dA_cumsum_chunk[i - chunk_start] = dt_chunk[i - chunk_start] * A_val
        if i > chunk_start:
            dA_cumsum_chunk[i - chunk_start] += dA_cumsum_chunk[i - chunk_start - 1]

    # Store results
    tl.store(dA_cumsum_ptr + dA_cumsum_offset, dA_cumsum_chunk, mask=chunk_start + tl.arange(0, BLOCK_SIZE) < chunk_end)
    tl.store(dt_out_ptr + dt_out_offset, dt_chunk, mask=chunk_start + tl.arange(0, BLOCK_SIZE) < chunk_end)

import torch

def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=10.0):
    batch, seqlen, nheads = dt.shape
    num_chunks = (seqlen + chunk_size - 1) // chunk_size

    dA_cumsum = torch.zeros((batch, num_chunks, nheads), device=dt.device, dtype=dt.dtype)
    dt_out = torch.zeros_like(dt)

    grid = (batch * nheads * num_chunks,)

    if dt_bias is not None:
        dt_bias_ptr = dt_bias.data_ptr()
    else:
        dt_bias_ptr = None

    _chunk_cumsum_fwd_kernel[grid](
        dt.data_ptr(),
        A.data_ptr(),
        dA_cumsum.data_ptr(),
        dt_out.data_ptr(),
        dt_bias_ptr,
        dt_softplus,
        dt_limit,
        batch,
        seqlen,
        nheads,
        chunk_size,
        BLOCK_SIZE=chunk_size,
    )

    return dA_cumsum, dt_out
