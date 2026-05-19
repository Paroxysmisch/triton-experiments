import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, Out_ptr, Out_scale_ptr, Dest_loc_ptr,
    stride_k_b, stride_k_h, stride_k_s,  # K strides
    stride_out_b, stride_out_h, stride_out_s,  # Out strides
    stride_scale_b, stride_scale_h,  # Scale strides
    BLOCK_SIZE: tl.constexpr,
    size_s: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    bid = pid // size_s  # Batch * Head index
    sid = pid % size_s   # Sequence index

    # Compute batch and head indices
    batch = bid // BLOCK_SIZE
    head = bid % BLOCK_SIZE

    # Load destination index
    dest_idx = tl.load(Dest_loc_ptr + sid)

    # Compute base pointers
    k_base = K_ptr + batch * stride_k_b + head * stride_k_h + sid * stride_k_s
    out_base = Out_ptr + batch * stride_out_b + head * stride_out_h + dest_idx * stride_out_s
    scale_base = Out_scale_ptr + batch * stride_scale_b + head * stride_scale_h

    # Load input values
    x = tl.load(k_base)

    # Compute scale (max absolute value)
    x_abs = tl.abs(x)
    scale = tl.max(x_abs, axis=0) / 127.0
    scale = tl.where(scale == 0, 1.0, scale)

    # Quantize to int8
    x_quantized = tl.cast(x / scale, tl.int8)

    # Store quantized values and scale
    tl.store(out_base, x_quantized)
    tl.store(scale_base + dest_idx, scale)

def destindex_copy_quantize_kv(k: torch.Tensor, dest_loc: torch.Tensor):
    """
    Quantize K tensor and copy to output based on destination indices.
    
    Args:
        k: Input tensor of shape [batch, num_heads, seq_len, dim]
        dest_loc: Destination indices tensor of shape [seq_len]
        
    Returns:
        Tuple of (quantized_output, scales)
    """
    batch, num_heads, seq_len, dim = k.shape
    
    # Allocate output tensors
    out = torch.empty_like(k, dtype=torch.int8, device=k.device)
    out_scale = torch.empty((batch, num_heads, seq_len), dtype=torch.float32, device=k.device)
    
    # Launch kernel
    grid = (batch * num_heads * seq_len,)
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        k, out, out_scale, dest_loc,
        k.stride(0), k.stride(1), k.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        out_scale.stride(0), out_scale.stride(1),
        BLOCK_SIZE=num_heads,
        size_s=seq_len,
    )
    
    return out, out_scale
