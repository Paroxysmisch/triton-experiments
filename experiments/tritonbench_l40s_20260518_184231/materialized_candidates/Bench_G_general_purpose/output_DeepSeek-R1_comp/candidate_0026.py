import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(
    q_ptr,
    q_int8_ptr,
    q_scale_ptr,
    stride_bh_q,
    stride_s_q,
    stride_d_q,
    stride_bh_int8,
    stride_s_int8,
    stride_d_int8,
    S,
    D,
    BLKQ: tl.constexpr,
    num_blocks_s,
    num_blocks_d,
):
    # Get 3D program indices for batch_head, s_block, d_block
    batch_head = tl.program_id(0)
    s_block = tl.program_id(1)
    d_block = tl.program_id(2)
    
    # Compute starting offsets for S and D dimensions
    s_start = s_block * BLKQ
    d_start = d_block * BLKQ
    
    # Generate offsets for the block with masking for boundary checks
    s_offs = s_start + tl.arange(0, BLKQ)
    d_offs = d_start + tl.arange(0, BLKQ)
    s_mask = s_offs < S
    d_mask = d_offs < D
    mask = s_mask[:, None] & d_mask[None, :]
    
    # Compute pointers to the block in the input and output tensors
    q_block_ptr = q_ptr + batch_head * stride_bh_q + (s_offs[:, None] * stride_s_q) + (d_offs[None, :] * stride_d_q)
    q_int8_block_ptr = q_int8_ptr + batch_head * stride_bh_int8 + (s_offs[:, None] * stride_s_int8) + (d_offs[None, :] * stride_d_int8)
    
    # Load the data block, initializing out-of-bounds elements to 0
    data = tl.load(q_block_ptr, mask=mask, other=0.0)
    
    # Compute maximum absolute value in the block for scaling
    abs_data = tl.abs(data)
    max_abs = tl.max(abs_data)
    max_abs = tl.maximum(max_abs, 1e-8)  # Avoid division by zero
    scale = 127.0 / max_abs
    
    # Quantize the data to int8 with rounding and clamping
    quantized = tl.math.round(data * scale)
    quantized = tl.minimum(tl.maximum(quantized, -127), 127)
    quantized = quantized.to(tl.int8)
    
    # Store the quantized data and scaling factor
    tl.store(q_int8_block_ptr, quantized, mask=mask)
    scale_ptr = q_scale_ptr + batch_head * (num_blocks_s * num_blocks_d) + s_block * num_blocks_d + d_block
    tl.store(scale_ptr, scale.to(tl.float16))

@triton.jit
def k_kernel_per_block_int8(
    k_ptr,
    k_int8_ptr,
    k_scale_ptr,
    stride_bh_k,
    stride_s_k,
    stride_d_k,
    stride_bh_int8,
    stride_s_int8,
    stride_d_int8,
    S,
    D,
    BLKK: tl.constexpr,
    num_blocks_s,
    num_blocks_d,
):
    # Same implementation as q_kernel with BLKQ replaced by BLKK
    batch_head = tl.program_id(0)
    s_block = tl.program_id(1)
    d_block = tl.program_id(2)
    
    s_start = s_block * BLKK
    d_start = d_block * BLKK
    
    s_offs = s_start + tl.arange(0, BLKK)
    d_offs = d_start + tl.arange(0, BLKK)
    s_mask = s_offs < S
    d_mask = d_offs < D
    mask = s_mask[:, None] & d_mask[None, :]
    
    k_block_ptr = k_ptr + batch_head * stride_bh_k + (s_offs[:, None] * stride_s_k) + (d_offs[None, :] * stride_d_k)
    k_int8_block_ptr = k_int8_ptr + batch_head * stride_bh_int8 + (s_offs[:, None] * stride_s_int8) + (d_offs[None, :] * stride_d_int8)
    
    data = tl.load(k_block_ptr, mask=mask, other=0.0)
    
    abs_data = tl.abs(data)
    max_abs = tl.max(abs_data)
    max_abs = tl.maximum(max_abs, 1e-8)
    scale = 127.0 / max_abs
    
    quantized = tl.math.round(data * scale)
    quantized = tl.minimum(tl.maximum(quantized, -127), 127)
    quantized = quantized.to(tl.int8)
    
    tl.store(k_int8_block_ptr, quantized, mask=mask)
    scale_ptr = k_scale_ptr + batch_head * (num_blocks_s * num_blocks_d) + s_block * num_blocks_d + d_block
    tl.store(scale_ptr, scale.to(tl.float16))

def per_block_int8(q, k, BLKQ, BLKK):
    # Reshape inputs to 4D (batch*heads, seq_len, 1, d_model)
    q_reshape = q.view(q.size(0), q.size(1), -1, q.size(-1))  # (B, H, S, D)
    B, H, S, D = q_reshape.shape
    q_reshape = q_reshape.view(B*H, S, 1, D)
    
    k_reshape = k.view(k.size(0), k.size(1), -1, k.size(-1))  # (B, H, S_k, D_k)
    B_k, H_k, S_k, D_k = k_reshape.shape
    k_reshape = k_reshape.view(B_k*H_k, S_k, 1, D_k)
    
    # Initialize output tensors
    q_int8 = torch.empty_like(q_reshape, dtype=torch.int8)
    k_int8 = torch.empty_like(k_reshape, dtype=torch.int8)
    
    num_blocks_s_q = (S + BLKQ - 1) // BLKQ
    num_blocks_d_q = (D + BLKQ - 1) // BLKQ
    q_scale = torch.empty((B*H, num_blocks_s_q, num_blocks_d_q), device=q.device, dtype=torch.float16)
    
    num_blocks_s_k = (S_k + BLKK - 1) // BLKK
    num_blocks_d_k = (D_k + BLKK - 1) // BLKK
    k_scale = torch.empty((B*H, num_blocks_s_k, num_blocks_d_k), device=k.device, dtype=torch.float16)
    
    # Launch q kernel
    grid_q = (B*H, num_blocks_s_q, num_blocks_d_q)
    q_kernel_per_block_int8[grid_q](
        q_reshape, q_int8, q_scale,
        q_reshape.stride(0), q_reshape.stride(1), q_reshape.stride(3),
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(3),
        S, D,
        BLKQ,
        num_blocks_s_q, num_blocks_d_q,
    )
    
    # Launch k kernel
    grid_k = (B*H, num_blocks_s_k, num_blocks_d_k)
    k_kernel_per_block_int8[grid_k](
        k_reshape, k_int8, k_scale,
        k_reshape.stride(0), k_reshape.stride(1), k_reshape.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(3),
        S_k, D_k,
        BLKK,
        num_blocks_s_k, num_blocks_d_k,
    )
    
    return q_int8, k_int8, q_scale, k_scale
