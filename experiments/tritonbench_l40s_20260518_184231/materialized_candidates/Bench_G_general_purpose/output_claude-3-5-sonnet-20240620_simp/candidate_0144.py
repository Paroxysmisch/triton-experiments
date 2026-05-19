import torch
import triton
import triton.language as tl
import math

@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to tensors
    dt_ptr, A_ptr, dt_bias_ptr, dA_cumsum_ptr, dt_out_ptr,
    # Shapes and strides
    batch, seqlen, nheads,
    dt_batch_stride, dt_seq_stride, dt_head_stride,
    A_stride,
    dt_bias_batch_stride, dt_bias_seq_stride, dt_bias_head_stride,
    dA_cumsum_batch_stride, dA_cumsum_seq_stride, dA_cumsum_head_stride,
    dt_out_batch_stride, dt_out_seq_stride, dt_out_head_stride,
    # Configuration
    chunk_size: tl.constexpr,
    dt_softplus: tl.constexpr,
    dt_limit: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate indices
    batch_idx = pid // (nheads * (seqlen // chunk_size))
    head_idx = (pid % (nheads * (seqlen // chunk_size))) // (seqlen // chunk_size)
    chunk_idx = pid % (seqlen // chunk_size)
    
    # Calculate starting positions
    chunk_start = chunk_idx * chunk_size
    
    # Load block
    offs_seq = chunk_start + tl.arange(0, BLOCK_SIZE)
    mask = offs_seq < min(chunk_start + chunk_size, seqlen)
    
    # Load scaling factor A
    a = tl.load(A_ptr + head_idx * A_stride)
    
    # Calculate memory offsets
    dt_offs = batch_idx * dt_batch_stride + head_idx * dt_head_stride + offs_seq * dt_seq_stride
    
    # Load dt values
    dt = tl.load(dt_ptr + dt_offs, mask=mask)
    
    # Apply bias if provided
    if dt_bias_ptr is not None:
        dt_bias_offs = batch_idx * dt_bias_batch_stride + head_idx * dt_bias_head_stride + offs_seq * dt_bias_seq_stride
        dt = dt + tl.load(dt_bias_ptr + dt_bias_offs, mask=mask)
    
    # Apply softplus if required
    if dt_softplus:
        dt = tl.log(1 + tl.exp(dt))
    
    # Apply scaling and clamping
    dt = dt * a
    if dt_limit > 0:
        dt = tl.minimum(tl.maximum(dt, -dt_limit), dt_limit)
    
    # Store modified dt
    dt_out_offs = batch_idx * dt_out_batch_stride + head_idx * dt_out_head_stride + offs_seq * dt_out_seq_stride
    tl.store(dt_out_ptr + dt_out_offs, dt, mask=mask)
    
    # Compute cumsum
    cumsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        if i < chunk_size and offs_seq[i] < seqlen:
            cumsum[i] = cumsum[i-1] + dt[i] if i > 0 else dt[i]
    
    # Store cumsum results
    dA_cumsum_offs = batch_idx * dA_cumsum_batch_stride + head_idx * dA_cumsum_head_stride + offs_seq * dA_cumsum_seq_stride
    tl.store(dA_cumsum_ptr + dA_cumsum_offs, cumsum, mask=mask)

def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=0.0):
    """
    Wrapper function for chunk-wise cumulative sum.
    
    Args:
        dt: Input tensor of shape (batch, seqlen, nheads)
        A: Scaling factors of shape (nheads,)
        chunk_size: Size of each chunk for processing
        dt_bias: Optional bias tensor for dt
        dt_softplus: Boolean to apply softplus transformation
        dt_limit: Clamping limits for dt
    
    Returns:
        tuple: (dA_cumsum, dt_out)
    """
    batch, seqlen, nheads = dt.shape
    
    # Ensure chunk_size divides seqlen
    assert seqlen % chunk_size == 0, f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})"
    
    # Prepare output tensors
    dA_cumsum = torch.empty_like(dt)
    dt_out = torch.empty_like(dt)
    
    # Calculate grid size
    grid = (batch * nheads * (seqlen // chunk_size),)
    
    # Determine optimal block size
    BLOCK_SIZE = min(chunk_size, 1024)  # Maximum triton block size
    
    # Launch kernel
    _chunk_cumsum_fwd_kernel[grid](
        dt, A, dt_bias if dt_bias is not None else None,
        dA_cumsum, dt_out,
        batch, seqlen, nheads,
        dt.stride(0), dt.stride(1), dt.stride(2),
        A.stride(0),
        dt_bias.stride(0) if dt_bias is not None else 0,
        dt_bias.stride(1) if dt_bias is not None else 0,
        dt_bias.stride(2) if dt_bias is not None else 0,
        dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
        dt_out.stride(0), dt_out.stride(1), dt_out.stride(2),
        chunk_size=chunk_size,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dA_cumsum, dt_out
