import torch
import triton
import triton.language as tl
import warnings
from typing import Optional

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q, k, v, cos, sin, k_cache, v_cache, BLOCK_TABLES, context_lengths, x,
    q_token_stride, q_head_stride, k_token_stride, k_head_stride, head_dim_stride,
    cos_token_stride, cos_stride, kcb_stride, kch_stride, kcsplit_x_stride,
    kcs_stride, kcd_stride, vcb_stride, vch_stride, vcs_stride, vcd_stride,
    bts_stride, btb_stride, block_size, KV_GROUP_NUM: tl.constexpr, HEAD_DIM: tl.constexpr
):
    # Program IDs to identify the specific head and token being processed
    pid = tl.program_id(axis=0)
    q_token_id = tl.program_id(axis=1)

    # Compute the start index for the head and token
    q_start = pid * q_head_stride + q_token_id * q_token_stride
    k_start = pid * k_head_stride + q_token_id * k_token_stride
    v_start = pid * q_head_stride + q_token_id * q_token_stride

    # Load q, k, v slices
    q_ptrs = q + q_start + tl.arange(0, HEAD_DIM)
    k_ptrs = k + k_start + tl.arange(0, HEAD_DIM)
    v_ptrs = v + v_start + tl.arange(0, HEAD_DIM)

    q_vals = tl.load(q_ptrs)
    k_vals = tl.load(k_ptrs)
    v_vals = tl.load(v_ptrs)

    # Load cos and sin slices
    cos_ptrs = cos + q_token_id * cos_token_stride + tl.arange(0, HEAD_DIM)
    sin_ptrs = sin + q_token_id * cos_token_stride + tl.arange(0, HEAD_DIM)

    cos_vals = tl.load(cos_ptrs)
    sin_vals = tl.load(sin_ptrs)

    # Apply rotary embedding transformation
    q_half1, q_half2 = tl.split(q_vals, 2)
    k_half1, k_half2 = tl.split(k_vals, 2)

    q_rotated = tl.cat([q_half1 * cos_vals - q_half2 * sin_vals, q_half1 * sin_vals + q_half2 * cos_vals], 0)
    k_rotated = tl.cat([k_half1 * cos_vals - k_half2 * sin_vals, k_half1 * sin_vals + k_half2 * cos_vals], 0)

    # Store the results back to q
    tl.store(q_ptrs, q_rotated)

    # Update k_cache and v_cache
    k_cache_ptrs = k_cache + pid * kch_stride + q_token_id * kcb_stride + tl.arange(0, HEAD_DIM)
    v_cache_ptrs = v_cache + pid * vch_stride + q_token_id * vcb_stride + tl.arange(0, HEAD_DIM)

    tl.store(k_cache_ptrs, k_rotated)
    tl.store(v_cache_ptrs, v_vals)

def decoding_fused_rotary_embedding(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None, v_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None, kv_lengths: Optional[torch.Tensor] = None,
    use_new_kcache_layout: bool = False
):
    q_total_tokens, q_head_num, head_dim = q.shape
    k_head_num = k.size(1)
    x = head_dim
    kcsplit_x_stride, kcs_stride, kcd_stride = 0, k_cache.stride(2), k_cache.stride(3)

    if use_new_kcache_layout:
        assert (
            k_cache.dim() == 5
            and k_cache.shape[1] == v_cache.shape[1]
            and k_cache.shape[2] * k_cache.shape[4] == v_cache.shape[3]
        ), f"Invalid KCache shape {k_cache.shape} and VCache shape {v_cache.shape}"
        x = k_cache.size(-1)
        kcsplit_x_stride, kcs_stride, kcd_stride = k_cache.stride()[-3:]

    grid = (q_head_num, q_total_tokens)
    decoding_fused_rotary_embedding_kernel[grid](
        q, k, v, cos, sin, k_cache, v_cache, block_tables, kv_lengths, x,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), q.stride(2),
        cos.stride(0), cos.stride(1), k_cache.stride(0), k_cache.stride(1),
        kcsplit_x_stride, kcs_stride, kcd_stride, v_cache.stride(0),
        v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1), k_cache.size(-2),
        KV_GROUP_NUM=q_head_num // k_head_num, HEAD_DIM=head_dim,
        num_warps=16 if head_dim >= 512 else 8 if head_dim >= 256 else 4,
    )
    return
