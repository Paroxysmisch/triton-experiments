import triton
import triton.language as tl

@triton.jit
def _chunk_cumsum_fwd_kernel(
    dt_ptr,  # Pointer to input tensor (batch, seqlen, nheads)
    A_ptr,   # Pointer to scaling factors (nheads)
    dA_cumsum_ptr,  # Pointer to output cumulative sum tensor (batch, nheads, nchunks, chunk_size)
    dt_out_ptr,  # Pointer to output transformed tensor (batch, nheads, nchunks, chunk_size)
    dt_bias_ptr,  # Optional pointer to biases (nheads)
    dt_softplus,  # Boolean flag for softplus transformation
    dt_limit_min,  # Minimum clamping value for dt
    dt_limit_max,  # Maximum clamping value for dt
    batch,  # Batch size
    seqlen,  # Sequence length
    nheads,  # Number of heads
    chunk_size,  # Chunk size
    BLOCK_BATCH: tl.constexpr,  # Block size for batch dimension
    BLOCK_SEQ: tl.constexpr,  # Block size for sequence dimension
    BLOCK_HEAD: tl.constexpr  # Block size for head dimension
):
    # Compute the grid and block indices
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Compute the start and end indices for the current chunk
    start_idx = pid_chunk * chunk_size
    end_idx = tl.minimum(start_idx + chunk_size, seqlen)

    # Compute the base pointers for the current batch and head
    dt_base_ptr = dt_ptr + (pid_batch * seqlen * nheads + pid_head * seqlen)
    dA_cumsum_base_ptr = dA_cumsum_ptr + (pid_batch * nheads * (seqlen // chunk_size) * chunk_size + pid_head * (seqlen // chunk_size) * chunk_size + pid_chunk * chunk_size)
    dt_out_base_ptr = dt_out_ptr + (pid_batch * nheads * (seqlen // chunk_size) * chunk_size + pid_head * (seqlen // chunk_size) * chunk_size + pid_chunk * chunk_size)

    # Load the scaling factor for the current head
    A = tl.load(A_ptr + pid_head)

    # Load the bias for the current head if provided
    if dt_bias_ptr is not None:
        bias = tl.load(dt_bias_ptr + pid_head)
    else:
        bias = 0.0

    # Initialize the cumulative sum
    cumsum = 0.0

    # Iterate over the chunk
    for i in range(start_idx, end_idx):
        # Load the current value
        dt_val = tl.load(dt_base_ptr + i)

        # Apply bias if provided
        dt_val += bias

        # Apply softplus transformation if enabled
        if dt_softplus:
            dt_val = tl.log(1.0 + tl.exp(dt_val))

        # Apply clamping
        dt_val = tl.maximum(dt_limit_min, tl.minimum(dt_val, dt_limit_max))

        # Compute the cumulative sum
        cumsum += dt_val * A

        # Store the cumulative sum and the transformed value
        tl.store(dA_cumsum_base_ptr + (i - start_idx), cumsum)
        tl.store(dt_out_base_ptr + (i - start_idx), dt_val)

import torch

def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(-1e9, 1e9)):
    # Check input shapes
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,), "A must have shape (nheads,)"
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,), "dt_bias must have shape (nheads,)"

    # Compute the number of chunks
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    # Initialize output tensors
    dA_cumsum = torch.zeros((batch, nheads, nchunks, chunk_size), device=dt.device, dtype=dt.dtype)
    dt_out = torch.zeros((batch, nheads, nchunks, chunk_size), device=dt.device, dtype=dt.dtype)

    # Determine block sizes
    BLOCK_BATCH = 1
    BLOCK_SEQ = 1
    BLOCK_HEAD = 1

    # Compute grid dimensions
    grid = (batch, nheads, nchunks)

    # Launch the kernel
    _chunk_cumsum_fwd_kernel[grid](
        dt, A, dA_cumsum, dt_out, dt_bias, dt_softplus, dt_limit[0], dt_limit[1],
        batch, seqlen, nheads, chunk_size,
        BLOCK_BATCH, BLOCK_SEQ, BLOCK_HEAD
    )

    return dA_cumsum, dt_out
