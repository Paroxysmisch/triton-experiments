import torch
import triton
import triton.language as tl
import warnings
from typing import Optional

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q, k, v, cos, sin, k_cache, v_cache,
    BLOCK_TABLES, context_lengths, x,
    q_token_stride, q_head_stride,
    k_token_stride, k_head_stride,
    head_dim_stride, cos_token_stride, cos_stride,
    kcb_stride, kch_stride, kcsplit_x_stride, kcs_stride, kcd_stride,
    vcb_stride, vch_stride, vcs_stride, vcd_stride,
    bts_stride, btb_stride, block_size,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    # Get program ID
    head_id = tl.program_id(0)
    token_id = tl.program_id(1)

    # Compute kv head id
    kv_head_id = head_id // KV_GROUP_NUM

    # Load context length for the current token
    context_len = tl.load(context_lengths + token_id)

    # Compute offsets
    q_offset = head_id * q_head_stride + token_id * q_token_stride
    k_offset = kv_head_id * k_head_stride + token_id * k_token_stride
    cos_offset = token_id * cos_token_stride

    # Create pointers for q, k, v
    q_ptr = q + q_offset
    k_ptr = k + k_offset
    v_ptr = v + k_offset  # Assuming v has the same layout as k
    cos_ptr = cos + cos_offset
    sin_ptr = sin + cos_offset

    # Load q, k, v
    q_half = tl.load(q_ptr + tl.arange(0, HEAD_DIM))
    k_half = tl.load(k_ptr + tl.arange(0, HEAD_DIM))
    v_half = tl.load(v_ptr + tl.arange(0, HEAD_DIM))

    # Load cos and sin
    cos_half = tl.load(cos_ptr + tl.arange(0, HEAD_DIM))
    sin_half = tl.load(sin_ptr + tl.arange(0, HEAD_DIM))

    # Compute rotary embedding
    q_rot = tl.where(tl.arange(0, HEAD_DIM) < HEAD_DIM // 2,
                     q_half * cos_half + tl.roll(q_half, HEAD_DIM // 2) * sin_half,
                     q_half * cos_half - tl.roll(q_half, -HEAD_DIM // 2) * sin_half)
    
    k_rot = tl.where(tl.arange(0, HEAD_DIM) < HEAD_DIM // 2,
                     k_half * cos_half + tl.roll(k_half, HEAD_DIM // 2) * sin_half,
                     k_half * cos_half - tl.roll(k_half, -HEAD_DIM // 2) * sin_half)

    # Store rotated q back to global memory
    tl.store(q_ptr + tl.arange(0, HEAD_DIM), q_rot)

    # Compute cache indices
    block_table_idx = tl.load(BLOCK_TABLES + token_id * bts_stride + context_len // block_size * btb_stride)
    k_cache_block_offset = block_table_idx * kcb_stride
    v_cache_block_offset = block_table_idx * vcb_stride

    # Update k_cache and v_cache
    k_cache_offset = k_cache_block_offset + kv_head_id * kch_stride + (context_len % block_size) * kcs_stride
    v_cache_offset = v_cache_block_offset + kv_head_id * vch_stride + (context_len % block_size) * vcs_stride

    tl.store(k_cache + k_cache_offset + tl.arange(0, HEAD_DIM) * kcd_stride, k_rot)
    tl.store(v_cache + v_cache_offset + tl.arange(0, HEAD_DIM) * vcd_stride, v_half)

def decoding_fused_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
    use_new_kcache_layout: bool = False,
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
        q, k, v, cos, sin, k_cache, v_cache,
        block_tables, kv_lengths, x,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        q.stride(2), cos.stride(0), cos.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        kcsplit_x_stride, kcs_stride, kcd_stride,
        v_cache.stride(0), v_cache.stride(1),
        v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        k_cache.size(-2),
        KV_GROUP_NUM=q_head_num // k_head_num,
        HEAD_DIM=head_dim,
        num_warps=16 if head_dim >= 512 else 8 if head_dim >= 256 else 4,
    )
    return
