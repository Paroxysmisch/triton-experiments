import torch
import triton
import triton.language as tl
import math

@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_cumsum_ptr,
    # Matrix dimensions
    batch, seqlen, nheads, chunk_size,
    # Limits for clamping
    dt_min, dt_max,
    # Matrix strides
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head, stride_dt_bias_head,
    stride_dt_out_batch, stride_dt_out_chunk, stride_dt_out_head, stride_dt_out_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # Optional configurations
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    # Program ID
    pid_b = tl.program_id(axis=0)  # Batch
    pid_c = tl.program_id(axis=1)  # Chunk
    pid_h = tl.program_id(axis=2)  # Head

    # Compute pointer offsets
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    # Create ranges for heads and chunk positions
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)

    # Compute pointers for each element
    dt_ptrs = dt_ptr + offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize
    dA_cs_ptrs = dA_cumsum_ptr + offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize

    # Handle boundary conditions
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    mask = (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit)

    # Load input data
    dt = tl.load(dt_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Apply bias if present
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, 
                         mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]

    # Apply softplus if requested
    if DT_SOFTPLUS:
        dt = tl.log1p(tl.exp(dt))  # softplus implementation

    # Apply clamping
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where(mask, dt, 0.0)

    # Store transformed dt
    tl.store(dt_out_ptrs, dt, mask=mask)

    # Compute cumulative sum
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=mask)

def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, 
                     dt_limit=(0.0, float('inf'))):
    """
    Wrapper function for the forward chunk cumsum operation.
    
    Args:
        dt: Input tensor of shape (batch, seqlen, nheads)
        A: Scaling factors of shape (nheads,)
        chunk_size: Size of each chunk for processing
        dt_bias: Optional bias tensor of shape (nheads,)
        dt_softplus: Boolean flag for softplus activation
        dt_limit: Tuple of (min, max) values for clamping
    
    Returns:
        Tuple of (dA_cumsum, dt_out)
    """
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)

    # Calculate number of chunks
    nchunks = math.ceil(seqlen / chunk_size)

    # Prepare output tensors
    dt_out = torch.empty(batch, nheads, nchunks, chunk_size, 
                        device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, 
                           device=dt.device, dtype=torch.float32)

    # Grid for kernel launch
    grid = (batch, nchunks, triton.cdiv(nheads, 32))  # 32 is default BLOCK_SIZE_H

    # Launch kernel
    _chunk_cumsum_fwd_kernel[grid](
        dt, A, dt_bias, dt_out, dA_cumsum,
        batch, seqlen, nheads, chunk_size,
        dt_limit[0], dt_limit[1],
        dt.stride(0), dt.stride(1), dt.stride(2),
        A.stride(0),
        dt_bias.stride(0) if dt_bias is not None else 0,
        dt_out.stride(0), dt_out.stride(2), dt_out.stride(1), dt_out.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        DT_SOFTPLUS=dt_softplus,
        HAS_DT_BIAS=dt_bias is not None,
        BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        BLOCK_SIZE_H=32,
    )

    return dA_cumsum, dt_out
