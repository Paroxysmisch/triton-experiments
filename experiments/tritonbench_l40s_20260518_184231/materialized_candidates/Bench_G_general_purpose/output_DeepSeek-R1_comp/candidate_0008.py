import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, DestLoc_ptr, Out_ptr, Out_scale_ptr,
    K_batch_stride, K_head_stride, K_dim_stride,
    DestLoc_batch_stride, DestLoc_seq_stride,
    Out_seq_stride, Out_head_stride, Out_group_stride, Out_dim_stride,
    Out_scale_seq_stride, Out_scale_head_stride, Out_scale_group_stride,
    head_dim, group_size,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    group_idx = tl.program_id(2)
    
    dest_index = tl.load(DestLoc_ptr + seq_idx * DestLoc_seq_stride)
    
    group_start = group_idx * group_size
    offsets = group_start + tl.arange(0, BLOCK_GROUP_DIM)
    mask = offsets < head_dim
    
    k_ptr = K_ptr + seq_idx * K_batch_stride + head_idx * K_head_stride + offsets * K_dim_stride
    k = tl.load(k_ptr, mask=mask, other=0.0)
    
    abs_k = tl.abs(k)
    max_val = tl.max(abs_k, axis=0)
    scale = tl.where(max_val > 0, max_val / 127.0, 1.0)
    
    quantized = tl.math.round(k / scale).to(tl.int8)
    
    out_ptr = Out_ptr + dest_index * Out_seq_stride + head_idx * Out_head_stride + group_idx * Out_group_stride
    out_ptrs = out_ptr + tl.arange(0, BLOCK_GROUP_DIM) * Out_dim_stride
    tl.store(out_ptrs, quantized, mask=mask)
    
    scale_ptr = Out_scale_ptr + dest_index * Out_scale_seq_stride + head_idx * Out_scale_head_stride + group_idx * Out_scale_group_stride
    tl.store(scale_ptr, scale)

def destindex_copy_quantize_kv(K: torch.Tensor, DestLoc: torch.Tensor, Out: torch.Tensor, Out_scale: torch.Tensor, group_size: int):
    assert K.dim() == 3, "K must be 3D (batch, head, head_dim)"
    batch, seqlen = DestLoc.shape
    num_heads, head_dim = K.shape[1], K.shape[2]
    assert head_dim % group_size == 0, "head_dim must be divisible by group_size"
    num_groups = head_dim // group_size
    
    K_expanded = K.unsqueeze(1).expand(-1, seqlen, -1, -1).contiguous()
    total_seqlen = batch * seqlen
    K_reshaped = K_expanded.view(total_seqlen, num_heads, head_dim)
    DestLoc_flat = DestLoc.view(-1).contiguous()
    
    grid = (total_seqlen, num_heads, num_groups)
    
    BLOCK_GROUP_DIM = group_size
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K_reshaped, DestLoc_flat, Out, Out_scale,
        K_reshaped.stride(0), K_reshaped.stride(1), K_reshaped.stride(2),
        DestLoc.stride(0), DestLoc.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Out_scale.stride(0), Out_scale.stride(1), Out_scale.stride(2),
        head_dim, group_size,
        BLOCK_GROUP_DIM=BLOCK_GROUP_DIM
    )
