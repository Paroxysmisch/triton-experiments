import triton
import triton.language as tl

@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt_ptr, A_ptr, dA_cumsum_ptr, dt_out_ptr,
    dt_bias_ptr, dt_softplus, dt_limit_min, dt_limit_max,
    batch, seqlen, nheads, chunk_size,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for each dimension
    pid_batch = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_head = tl.program_id(2)

    # Compute the start index for this program
    offset_dt = pid_batch * seqlen * nheads + pid_head * seqlen + pid_chunk * chunk_size
    offset_dA_cumsum = pid_batch * nheads * (seqlen // chunk_size) * chunk_size + pid_head * (seqlen // chunk_size) * chunk_size + pid_chunk * chunk_size

    # Load scaling factor for the current head
    A = tl.load(A_ptr + pid_head)

    # Initialize cumulative sum
    cumsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the chunk
    for i in range(0, chunk_size, BLOCK_SIZE):
        # Load a block of data
        dt_block = tl.load(dt_ptr + offset_dt + i + tl.arange(0, BLOCK_SIZE))

        # Apply optional bias
        if dt_bias_ptr is not None:
            bias = tl.load(dt_bias_ptr + pid_head)
            dt_block += bias

        # Apply softplus transformation if required
        if dt_softplus:
            dt_block = tl.log1p(tl.exp(dt_block))

        # Clamp the values
        dt_block = tl.clamp(dt_block, dt_limit_min, dt_limit_max)

        # Store the transformed dt
        tl.store(dt_out_ptr + offset_dA_cumsum + i + tl.arange(0, BLOCK_SIZE), dt_block)

        # Update cumulative sum
        cumsum += A * dt_block

        # Store cumulative sum result
        tl.store(dA_cumsum_ptr + offset_dA_cumsum + i + tl.arange(0, BLOCK_SIZE), cumsum)

def _chunk_cumsum_fwd(
    dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(-float('inf'), float('inf'))
):
    batch, seqlen, nheads = dt.shape
    nchunks = seqlen // chunk_size

    # Initialize output tensors
    dA_cumsum = torch.empty((batch, nheads, nchunks, chunk_size), device=dt.device, dtype=dt.dtype)
    dt_out = torch.empty_like(dA_cumsum)

    # Grid dimensions
    grid = (batch, nchunks, nheads)

    # Launch the kernel
    _chunk_cumsum_fwd_kernel[grid](
        dt_ptr=dt, A_ptr=A, dA_cumsum_ptr=dA_cumsum, dt_out_ptr=dt_out,
        dt_bias_ptr=dt_bias, dt_softplus=dt_softplus,
        dt_limit_min=dt_limit[0], dt_limit_max=dt_limit[1],
        batch=batch, seqlen=seqlen, nheads=nheads, chunk_size=chunk_size,
        BLOCK_SIZE=128  # Adjust based on your hardware capabilities
    )

    return dA_cumsum, dt_out
