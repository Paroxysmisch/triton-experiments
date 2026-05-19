import triton
import triton.language as tl
import torch

@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to tensors
    dt_ptr, A_ptr, dA_cumsum_ptr, dt_out_ptr, dt_bias_ptr,
    # Dimensions
    batch_size, seqlen, nheads, chunk_size,
    # Optional parameters
    dt_softplus: tl.constexpr, dt_limit_min: tl.constexpr, dt_limit_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate indices
    batch_idx = pid // (nheads * (seqlen // chunk_size))
    head_idx = (pid // (seqlen // chunk_size)) % nheads
    chunk_idx = pid % (seqlen // chunk_size)
    
    # Offsets for the current chunk
    chunk_start = chunk_idx * chunk_size
    
    # Load scaling factor A for current head
    A = tl.load(A_ptr + head_idx)
    
    # Load bias if provided
    bias = 0.0
    if dt_bias_ptr is not None:
        bias = tl.load(dt_bias_ptr + head_idx)
    
    # Initialize cumsum for the chunk
    cumsum = 0.0
    
    # Process chunk with BLOCK_SIZE elements at a time
    offs = tl.arange(0, BLOCK_SIZE)
    for i in range(0, chunk_size, BLOCK_SIZE):
        mask = offs + i < chunk_size
        
        # Load dt values
        dt_idx = batch_idx * seqlen * nheads + (chunk_start + i + offs) * nheads + head_idx
        dt = tl.load(dt_ptr + dt_idx, mask=mask)
        
        # Apply bias and transformations
        dt = dt + bias
        if dt_softplus:
            dt = tl.log(1.0 + tl.exp(dt))
        dt = tl.clamp(dt, dt_limit_min, dt_limit_max)
        
        # Store transformed dt
        dt_out_idx = (batch_idx * nheads * seqlen + head_idx * seqlen + chunk_start + i + offs)
        tl.store(dt_out_ptr + dt_out_idx, dt, mask=mask)
        
        # Compute cumsum
        dt = dt * A
        cumsum += tl.sum(dt * mask)
        
        # Store cumsum
        cumsum_idx = (batch_idx * nheads * (seqlen // chunk_size) * chunk_size + 
                      head_idx * (seqlen // chunk_size) * chunk_size +
                      chunk_idx * chunk_size + i + offs)
        tl.store(dA_cumsum_ptr + cumsum_idx, cumsum, mask=mask)

def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(-5.0, 5.0)):
    batch, seqlen, nheads = dt.shape
    device = dt.device
    
    # Input validation
    assert seqlen % chunk_size == 0, "Sequence length must be divisible by chunk size"
    assert A.shape == (nheads,), "A must have shape (nheads,)"
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,), "dt_bias must have shape (nheads,)"
    
    # Initialize output tensors
    nchunks = seqlen // chunk_size
    dA_cumsum = torch.empty((batch, nheads, nchunks, chunk_size), device=device, dtype=dt.dtype)
    dt_out = torch.empty_like(dt)
    
    # Determine grid and block sizes
    BLOCK_SIZE = min(chunk_size, 1024)
    grid = (batch * nheads * nchunks,)
    
    # Launch kernel
    _chunk_cumsum_fwd_kernel[grid](
        dt, A, dA_cumsum, dt_out, dt_bias,
        batch, seqlen, nheads, chunk_size,
        dt_softplus, dt_limit[0], dt_limit[1],
        BLOCK_SIZE,
    )
    
    return dA_cumsum, dt_out
