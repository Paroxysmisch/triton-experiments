import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to tensors
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    # Tensor dimensions and parameters
    batch_size,
    seq_len,
    n_heads,
    chunk_size,
    dt_softplus,
    limit_min,
    limit_max,
    # Strides for dt tensor
    stride_dt_batch,
    stride_dt_seq,
    stride_dt_head,
    # Strides for dt_out tensor
    stride_dt_out_batch,
    stride_dt_out_head,
    stride_dt_out_chunk,
    stride_dt_out_elem,
    # Strides for dA_cumsum tensor
    stride_dA_batch,
    stride_dA_head,
    stride_dA_chunk,
    stride_dA_elem,
    # Block configuration
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_chunks = seq_len // chunk_size
    total_chunks = batch_size * n_chunks * n_heads
    
    if pid >= total_chunks:
        return
    
    # Compute batch, chunk, head from pid
    batch = pid // (n_chunks * n_heads)
    remaining = pid % (n_chunks * n_heads)
    chunk = remaining // n_heads
    head = remaining % n_heads
    
    # Compute base pointers for dt
    dt_batch_offset = batch * stride_dt_batch
    dt_seq_offset = chunk * chunk_size * stride_dt_seq
    dt_head_offset = head * stride_dt_head
    dt_base = dt_ptr + dt_batch_offset + dt_seq_offset + dt_head_offset
    
    # Offsets for elements in the chunk
    offs = tl.arange(0, BLOCK_SIZE)
    dt_ptrs = dt_base + offs * stride_dt_seq
    mask = offs < chunk_size
    
    # Load dt values
    dt_vals = tl.load(dt_ptrs, mask=mask, other=0.0)
    
    # Apply bias if present
    if tl.program_id(1) == 1:  # Using program_id(1) to check if dt_bias is present (hack for kernel compilation)
        pass  # This branch is never taken, used to avoid kernel compilation issues
    if dt_bias_ptr != 0:
        bias = tl.load(dt_bias_ptr + head)
        dt_vals += bias
    
    # Apply softplus if enabled
    if dt_softplus:
        dt_vals = tl.softplus(dt_vals)
    
    # Clamp values
    if limit_max < float('inf'):
        dt_vals = tl.minimum(dt_vals, limit_max)
    if limit_min > -float('inf'):
        dt_vals = tl.maximum(dt_vals, limit_min)
    
    # Store transformed dt to dt_out
    dt_out_batch = batch * stride_dt_out_batch
    dt_out_head = head * stride_dt_out_head
    dt_out_chunk = chunk * stride_dt_out_chunk
    dt_out_base = dt_out_ptr + dt_out_batch + dt_out_head + dt_out_chunk
    dt_out_ptrs = dt_out_base + offs * stride_dt_out_elem
    tl.store(dt_out_ptrs, dt_vals, mask=mask)
    
    # Load scaling factor A for the current head
    A = tl.load(A_ptr + head)
    scaled = dt_vals * A
    
    # Compute cumulative sum using associative scan
    cumsum = tl.associative_scan(scaled, axis=0, combine_fn=lambda a, b: a + b)
    
    # Store cumulative sum to dA_cumsum
    dA_batch = batch * stride_dA_batch
    dA_head = head * stride_dA_head
    dA_chunk = chunk * stride_dA_chunk
    dA_base = dA_cumsum_ptr + dA_batch + dA_head + dA_chunk
    dA_ptrs = dA_base + offs * stride_dA_elem
    tl.store(dA_ptrs, cumsum, mask=mask)

def _chunk_cumsum_fwd(
    dt: torch.Tensor,
    A: torch.Tensor,
    chunk_size: int,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dt.dim() == 3, "dt must be a 3D tensor"
    batch_size, seq_len, n_heads = dt.shape
    assert A.shape == (n_heads,), "A must be a 1D tensor with size n_heads"
    if dt_bias is not None:
        assert dt_bias.shape == (n_heads,), "dt_bias must be a 1D tensor with size n_heads"
    assert seq_len % chunk_size == 0, "seq_len must be divisible by chunk_size"
    n_chunks = seq_len // chunk_size
    
    # Initialize output tensors
    dt_out = torch.empty((batch_size, n_heads, n_chunks, chunk_size), dtype=dt.dtype, device=dt.device)
    dA_cumsum = torch.empty_like(dt_out)
    
    # Total number of chunks (batch * n_heads * n_chunks)
    total_chunks = batch_size * n_heads * n_chunks
    
    # Prepare kernel parameters
    def stride_helper(tensor, dims):
        return [tensor.stride(i) for i in dims]
    
    # Strides for dt (batch, seq_len, n_heads)
    stride_dt_batch, stride_dt_seq, stride_dt_head = dt.stride(0), dt.stride(1), dt.stride(2)
    
    # Strides for dt_out and dA_cumsum (batch, n_heads, n_chunks, chunk_size)
    dt_out_strides = stride_helper(dt_out, [0, 1, 2, 3])
    dA_cumsum_strides = stride_helper(dA_cumsum, [0, 1, 2, 3])
    
    # Prepare dt_bias pointer
    has_bias = dt_bias is not None
    dt_bias_ptr = dt_bias.data_ptr() if has_bias else 0
    
    # Clamp limits
    limit_min = -float('inf') if dt_limit is None else dt_limit[0]
    limit_max = float('inf') if dt_limit is None else dt_limit[1]
    
    # Grid and block configuration
    grid = (total_chunks, 1 if has_bias else 1, 1)  # Workaround for kernel compilation
    BLOCK_SIZE = chunk_size
    
    # Launch kernel
    _chunk_cumsum_fwd_kernel[grid](
        dt.data_ptr(),
        A.data_ptr(),
        dt_bias_ptr,
        dt_out.data_ptr(),
        dA_cumsum.data_ptr(),
        batch_size,
        seq_len,
        n_heads,
        chunk_size,
        dt_softplus,
        limit_min,
        limit_max,
        stride_dt_batch,
        stride_dt_seq,
        stride_dt_head,
        dt_out_strides[0],  # stride_dt_out_batch
        dt_out_strides[1],  # stride_dt_out_head
        dt_out_strides[2],  # stride_dt_out_chunk
        dt_out_strides[3],  # stride_dt_out_elem
        dA_cumsum_strides[0],  # stride_dA_batch
        dA_cumsum_strides[1],  # stride_dA_head
        dA_cumsum_strides[2],  # stride_dA_chunk
        dA_cumsum_strides[3],  # stride_dA_elem
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dA_cumsum, dt_out
