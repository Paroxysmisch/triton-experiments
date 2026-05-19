import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(
    q_ptr,          # pointer to query matrix (float)
    q_int8_ptr,     # pointer to output quantized query matrix (int8)
    q_scale_ptr,    # pointer to output scale factors
    stride_qm,      # stride for q matrix rows
    stride_qk,      # stride for q matrix columns
    n_rows,         # number of rows
    n_cols,         # number of columns
    BLOCK_SIZE: tl.constexpr,  # block size for processing
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start indices
    block_start_row = pid * BLOCK_SIZE
    
    # Create block pointers
    offs_m = block_start_row + tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, n_cols)
    
    # Create mask for valid elements
    mask = offs_m < n_rows
    
    # Load block data
    q_block = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk, 
                      mask=mask[:, None], other=0.0)
    
    # Calculate scale (max absolute value in block)
    q_abs_max = tl.max(tl.abs(q_block), axis=1)
    scale = q_abs_max / 127.0  # Scale to use full int8 range
    
    # Quantize to int8
    q_scaled = q_block / scale[:, None]
    q_int8 = tl.math.round(q_scaled).to(tl.int8)
    
    # Store results
    tl.store(q_int8_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk,
             q_int8, mask=mask[:, None])
    tl.store(q_scale_ptr + offs_m, scale, mask=mask)

@triton.jit
def k_kernel_per_block_int8(
    k_ptr,          # pointer to key matrix (float)
    k_int8_ptr,     # pointer to output quantized key matrix (int8)
    k_scale_ptr,    # pointer to output scale factors
    stride_km,      # stride for k matrix rows
    stride_kk,      # stride for k matrix columns
    n_rows,         # number of rows
    n_cols,         # number of columns
    BLOCK_SIZE: tl.constexpr,  # block size for processing
):
    # Implementation similar to q_kernel_per_block_int8
    pid = tl.program_id(0)
    block_start_row = pid * BLOCK_SIZE
    
    offs_m = block_start_row + tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, n_cols)
    
    mask = offs_m < n_rows
    
    k_block = tl.load(k_ptr + offs_m[:, None] * stride_km + offs_n[None, :] * stride_kk,
                      mask=mask[:, None], other=0.0)
    
    k_abs_max = tl.max(tl.abs(k_block), axis=1)
    scale = k_abs_max / 127.0
    
    k_scaled = k_block / scale[:, None]
    k_int8 = tl.math.round(k_scaled).to(tl.int8)
    
    tl.store(k_int8_ptr + offs_m[:, None] * stride_km + offs_n[None, :] * stride_kk,
             k_int8, mask=mask[:, None])
    tl.store(k_scale_ptr + offs_m, scale, mask=mask)

def per_block_int8(q: torch.Tensor, k: torch.Tensor, BLKQ: int = 16, BLKK: int = 16):
    """
    Convert query and key matrices to int8 format with per-block scaling
    
    Args:
        q: Query matrix (batch_size, seq_len, dim)
        k: Key matrix (batch_size, seq_len, dim)
        BLKQ: Block size for query processing
        BLKK: Block size for key processing
    
    Returns:
        Tuple of (q_int8, q_scale, k_int8, k_scale)
    """
    batch_size, seq_len, dim = q.shape
    
    # Prepare output tensors
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    k_int8 = torch.empty_like(k, dtype=torch.int8)
    
    q_scale = torch.empty((batch_size, seq_len), dtype=torch.float32, device=q.device)
    k_scale = torch.empty((batch_size, seq_len), dtype=torch.float32, device=k.device)
    
    # Calculate grid sizes
    grid_q = (seq_len + BLKQ - 1) // BLKQ
    grid_k = (seq_len + BLKK - 1) // BLKK
    
    # Launch kernels for each batch
    for b in range(batch_size):
        # Process query matrix
        q_kernel_per_block_int8[(grid_q,)](
            q[b].contiguous().data_ptr(),
            q_int8[b].contiguous().data_ptr(),
            q_scale[b].contiguous().data_ptr(),
            q[b].stride(0),
            q[b].stride(1),
            seq_len,
            dim,
            BLOCK_SIZE=BLKQ,
        )
        
        # Process key matrix
        k_kernel_per_block_int8[(grid_k,)](
            k[b].contiguous().data_ptr(),
            k_int8[b].contiguous().data_ptr(),
            k_scale[b].contiguous().data_ptr(),
            k[b].stride(0),
            k[b].stride(1),
            seq_len,
            dim,
            BLOCK_SIZE=BLKK,
        )
    
    return q_int8, q_scale, k_int8, k_scale
