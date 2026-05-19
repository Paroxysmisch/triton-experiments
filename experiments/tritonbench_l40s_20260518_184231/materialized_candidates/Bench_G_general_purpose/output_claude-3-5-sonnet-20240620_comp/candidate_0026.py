import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(
    q_ptr,          # pointer to query matrix
    q_int8_ptr,     # pointer to output int8 matrix
    q_scale_ptr,    # pointer to scaling factors
    stride_qm,      # stride for q matrix rows
    stride_qk,      # stride for q matrix columns
    stride_qint8m,  # stride for output int8 matrix rows
    stride_qint8k,  # stride for output int8 matrix columns
    stride_qscale,  # stride for scaling factors
    nheads,         # number of attention heads
    seqlen,         # sequence length
    BLOCK_SIZE: tl.constexpr  # block size for processing
):
    # Get program ID for parallel execution
    pid = tl.program_id(0)
    
    # Calculate block indices
    head_idx = pid // (seqlen // BLOCK_SIZE)
    block_idx = pid % (seqlen // BLOCK_SIZE)
    
    # Calculate offsets
    offset_q = head_idx * stride_qm + block_idx * BLOCK_SIZE * stride_qk
    offset_q_int8 = head_idx * stride_qint8m + block_idx * BLOCK_SIZE * stride_qint8k
    offset_scale = head_idx * stride_qscale + block_idx * BLOCK_SIZE
    
    # Create block pointers
    q_block_ptr = q_ptr + offset_q
    q_int8_block_ptr = q_int8_ptr + offset_q_int8
    q_scale_block_ptr = q_scale_ptr + offset_scale
    
    # Load block data
    q_block = tl.load(q_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_qk + 
                     tl.arange(0, BLOCK_SIZE)[None, :])
    
    # Find max absolute value for scaling
    q_abs_max = tl.max(tl.abs(q_block))
    scale = q_abs_max / 127.0  # Scale to int8 range
    
    # Quantize to int8
    q_int8_block = tl.round(q_block / scale).to(tl.int8)
    
    # Store results
    tl.store(q_int8_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_qint8k + 
             tl.arange(0, BLOCK_SIZE)[None, :], q_int8_block)
    tl.store(q_scale_block_ptr + tl.arange(0, BLOCK_SIZE), scale)

@triton.jit
def k_kernel_per_block_int8(
    k_ptr,          # pointer to key matrix
    k_int8_ptr,     # pointer to output int8 matrix
    k_scale_ptr,    # pointer to scaling factors
    stride_km,      # stride for k matrix rows
    stride_kk,      # stride for k matrix columns
    stride_kint8m,  # stride for output int8 matrix rows
    stride_kint8k,  # stride for output int8 matrix columns
    stride_kscale,  # stride for scaling factors
    nheads,         # number of attention heads
    seqlen,         # sequence length
    BLOCK_SIZE: tl.constexpr  # block size for processing
):
    # Implementation similar to q_kernel_per_block_int8
    pid = tl.program_id(0)
    
    head_idx = pid // (seqlen // BLOCK_SIZE)
    block_idx = pid % (seqlen // BLOCK_SIZE)
    
    offset_k = head_idx * stride_km + block_idx * BLOCK_SIZE * stride_kk
    offset_k_int8 = head_idx * stride_kint8m + block_idx * BLOCK_SIZE * stride_kint8k
    offset_scale = head_idx * stride_kscale + block_idx * BLOCK_SIZE
    
    k_block_ptr = k_ptr + offset_k
    k_int8_block_ptr = k_int8_ptr + offset_k_int8
    k_scale_block_ptr = k_scale_ptr + offset_scale
    
    k_block = tl.load(k_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_kk + 
                     tl.arange(0, BLOCK_SIZE)[None, :])
    
    k_abs_max = tl.max(tl.abs(k_block))
    scale = k_abs_max / 127.0
    
    k_int8_block = tl.round(k_block / scale).to(tl.int8)
    
    tl.store(k_int8_block_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_kint8k + 
             tl.arange(0, BLOCK_SIZE)[None, :], k_int8_block)
    tl.store(k_scale_block_ptr + tl.arange(0, BLOCK_SIZE), scale)

def per_block_int8(q, k, BLOCK_SIZE=32):
    """
    Wrapper function to quantize query and key matrices to int8
    
    Args:
        q: Query tensor of shape [batch, nheads, seqlen, headdim]
        k: Key tensor of shape [batch, nheads, seqlen, headdim]
        BLOCK_SIZE: Size of processing blocks
    
    Returns:
        q_int8, k_int8: Quantized tensors
        q_scale, k_scale: Scaling factors
    """
    batch, nheads, seqlen, headdim = q.shape
    
    # Reshape inputs for processing
    q = q.reshape(-1, seqlen, headdim)
    k = k.reshape(-1, seqlen, headdim)
    
    # Initialize output tensors
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    k_int8 = torch.empty_like(k, dtype=torch.int8)
    q_scale = torch.empty((batch * nheads, seqlen), dtype=torch.float32, device=q.device)
    k_scale = torch.empty((batch * nheads, seqlen), dtype=torch.float32, device=k.device)
    
    # Calculate grid size
    grid = (nheads * (seqlen // BLOCK_SIZE),)
    
    # Launch kernels
    q_kernel_per_block_int8[grid](
        q, q_int8, q_scale,
        q.stride(0), q.stride(1),
        q_int8.stride(0), q_int8.stride(1),
        q_scale.stride(0),
        nheads, seqlen,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    k_kernel_per_block_int8[grid](
        k, k_int8, k_scale,
        k.stride(0), k.stride(1),
        k_int8.stride(0), k_int8.stride(1),
        k_scale.stride(0),
        nheads, seqlen,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape outputs back to original dimensions
    q_int8 = q_int8.reshape(batch, nheads, seqlen, headdim)
    k_int8 = k_int8.reshape(batch, nheads, seqlen, headdim)
    q_scale = q_scale.reshape(batch, nheads, seqlen)
    k_scale = k_scale.reshape(batch, nheads, seqlen)
    
    return q_int8, k_int8, q_scale, k_scale
