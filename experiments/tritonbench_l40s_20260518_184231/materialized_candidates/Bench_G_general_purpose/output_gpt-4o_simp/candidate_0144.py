import triton
import triton.language as tl

@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_cumsum_ptr,
    chunk_size, dt_softplus, dt_limit,
    batch, seqlen, nheads,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the index of the current block
    pid = tl.program_id(axis=0)
    
    # Compute offsets for each dimension
    batch_id = pid // (seqlen * nheads)
    seqlen_id = (pid // nheads) % seqlen
    head_id = pid % nheads

    # Calculate the start and end index for the current chunk
    chunk_start = seqlen_id * chunk_size
    chunk_end = min(chunk_start + chunk_size, seqlen)

    # Load scaling factor for the current head
    A = tl.load(A_ptr + head_id)

    # Initialize cumulative sum
    cumsum = 0.0

    # Iterate over the chunk and compute cumulative sum
    for i in range(chunk_start, chunk_end):
        # Load the current value
        idx = batch_id * seqlen * nheads + i * nheads + head_id
        dt_value = tl.load(dt_ptr + idx)

        # Apply optional bias
        if dt_bias_ptr:
            bias = tl.load(dt_bias_ptr + idx)
            dt_value += bias

        # Apply scaling
        dt_value *= A

        # Apply optional softplus
        if dt_softplus:
            dt_value = tl.log(1 + tl.exp(dt_value))

        # Apply clamping
        dt_value = tl.clamp(dt_value, dt_limit[0], dt_limit[1])

        # Update cumulative sum
        cumsum += dt_value

        # Store the modified dt value
        tl.store(dt_out_ptr + idx, dt_value)

    # Store the cumulative sum result
    dA_cumsum_idx = batch_id * nheads + head_id
    tl.store(dA_cumsum_ptr + dA_cumsum_idx, cumsum)

import torch

def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(-float('inf'), float('inf'))):
    batch, seqlen, nheads = dt.shape

    # Allocate output tensors
    dt_out = torch.empty_like(dt)
    dA_cumsum = torch.empty((batch, nheads), dtype=dt.dtype, device=dt.device)

    # Define the number of blocks
    num_blocks = batch * seqlen * nheads

    # Launch the kernel
    grid = (num_blocks,)
    _chunk_cumsum_fwd_kernel[grid](
        dt_ptr=dt,
        A_ptr=A,
        dt_bias_ptr=dt_bias,
        dt_out_ptr=dt_out,
        dA_cumsum_ptr=dA_cumsum,
        chunk_size=chunk_size,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        batch=batch,
        seqlen=seqlen,
        nheads=nheads,
        BLOCK_SIZE=chunk_size
    )

    return dA_cumsum, dt_out
