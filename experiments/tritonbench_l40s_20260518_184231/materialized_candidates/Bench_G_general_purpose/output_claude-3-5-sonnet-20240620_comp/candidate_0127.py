import torch
import triton
import triton.language as tl

# Triton kernel for quantizing and copying KV cache with destination indices
@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, Dest_loc_ptr, Out_ptr, Out_scale_ptr,
    stride_k_batch, stride_k_head, stride_k_seq, stride_k_dmodel,
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dmodel,
    seq_len, num_heads, dmodel,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)

    # Initialize offsets
    offs_head = pid_head * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    offs_dmodel = tl.arange(0, BLOCK_DMODEL)
    
    # Compute input offsets
    k_offs = (pid_batch * stride_k_batch + 
              offs_head[:, None] * stride_k_head +
              pid_seq * stride_k_seq +
              offs_dmodel[None, :] * stride_k_dmodel)
    
    # Load destination index
    dest_idx = tl.load(Dest_loc_ptr + pid_seq)
    
    # Load input block
    mask = (offs_head[:, None] < num_heads) & (offs_dmodel[None, :] < dmodel)
    x = tl.load(K_ptr + k_offs, mask=mask, other=0.0)
    
    # Compute scale (maximum absolute value) for the block
    x_abs = tl.abs(x)
    x_max = tl.max(x_abs, axis=1)[:, None]
    scale = x_max / 127.0
    
    # Quantize to int8
    x_scaled = x / scale
    x_int8 = tl.math.round(x_scaled)
    x_int8 = tl.math.min(127, tl.math.max(-127, x_int8))
    
    # Compute output offsets using destination index
    out_offs = (pid_batch * stride_out_batch +
                offs_head[:, None] * stride_out_head +
                dest_idx * stride_out_seq +
                offs_dmodel[None, :] * stride_out_dmodel)
    
    # Store quantized values and scales
    tl.store(Out_ptr + out_offs, x_int8, mask=mask)
    tl.store(Out_scale_ptr + pid_batch * num_heads + offs_head, scale.squeeze(1), 
             mask=offs_head < num_heads)

# PyTorch wrapper function
@torch.no_grad()
def destindex_copy_quantize_kv(k: torch.Tensor, dest_loc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize and copy KV cache with destination indices.
    
    Args:
        k: Input tensor of shape (batch_size, num_heads, seq_len, dmodel)
        dest_loc: Destination indices tensor of shape (seq_len,)
    
    Returns:
        tuple: (quantized_output, scales)
    """
    batch_size, num_heads, seq_len, dmodel = k.shape
    
    # Determine block sizes (power of 2)
    BLOCK_HEAD = triton.next_power_of_2(min(num_heads, 32))
    BLOCK_DMODEL = triton.next_power_of_2(min(dmodel, 64))
    
    # Initialize output tensors
    out = torch.empty_like(k, dtype=torch.int8, device=k.device)
    out_scale = torch.empty((batch_size, num_heads), dtype=k.dtype, device=k.device)
    
    # Compute strides
    stride_k_batch = k.stride(0)
    stride_k_head = k.stride(1)
    stride_k_seq = k.stride(2)
    stride_k_dmodel = k.stride(3)
    
    stride_out_batch = out.stride(0)
    stride_out_head = out.stride(1)
    stride_out_seq = out.stride(2)
    stride_out_dmodel = out.stride(3)
    
    # Launch kernel
    grid = (batch_size, triton.cdiv(num_heads, BLOCK_HEAD), seq_len)
    
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        k, dest_loc, out, out_scale,
        stride_k_batch, stride_k_head, stride_k_seq, stride_k_dmodel,
        stride_out_batch, stride_out_head, stride_out_seq, stride_out_dmodel,
        seq_len, num_heads, dmodel,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2
    )
    
    return out, out_scale
