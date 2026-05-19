import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q,
    k,
    v,
    cos,
    sin,
    k_cache,
    v_cache,
    BLOCK_TABLES,
    context_lengths,
    x,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    kcb_stride,
    kch_stride,
    kcsplit_x_stride,
    kcs_stride,
    kcd_stride,
    vcb_stride,
    vch_stride,
    vcs_stride,
    vcd_stride,
    bts_stride,
    btb_stride,
    block_size,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_0 = tl.program_id(0)  # Head index
    pid_1 = tl.program_id(1)  # Token index

    kv_head_idx = pid_0 // KV_GROUP_NUM
    is_first_in_group = (pid_0 % KV_GROUP_NUM) == 0

    pos_ptr = context_lengths + pid_1
    pos = tl.load(pos_ptr)
    cos_ptr = cos + pos * cos_token_stride
    sin_ptr = sin + pos * cos_token_stride

    # Process query
    q_ptr = q + pid_1 * q_token_stride + pid_0 * q_head_stride
    for i in range(0, HEAD_DIM // 2):
        q0 = tl.load(q_ptr + i * head_dim_stride)
        q1 = tl.load(q_ptr + (i + HEAD_DIM//2) * head_dim_stride)
        cos_val = tl.load(cos_ptr + i * cos_stride)
        sin_val = tl.load(sin_ptr + i * cos_stride)
        new_q0 = q0 * cos_val - q1 * sin_val
        new_q1 = q1 * cos_val + q0 * sin_val
        tl.store(q_ptr + i * head_dim_stride, new_q0)
        tl.store(q_ptr + (i + HEAD_DIM//2) * head_dim_stride, new_q1)

    if not is_first_in_group:
        return

    # Process key and value for cache
    k_ptr = k + pid_1 * k_token_stride + kv_head_idx * k_head_stride
    v_ptr = v + pid_1 * k_token_stride + kv_head_idx * k_head_stride

    batch_idx = pid_1
    current_length = tl.load(context_lengths + batch_idx)
    block_idx = current_length // block_size
    offset_in_block = current_length % block_size

    block_table_ptr = BLOCK_TABLES + batch_idx * btb_stride
    block_number = tl.load(block_table_ptr + block_idx * bts_stride)

    # Update key cache
    for i in tl.static_range(0, HEAD_DIM // 2):
        k0 = tl.load(k_ptr + i * head_dim_stride)
        k1 = tl.load(k_ptr + (i + HEAD_DIM//2) * head_dim_stride)
        cos_val = tl.load(cos_ptr + i * cos_stride)
        sin_val = tl.load(sin_ptr + i * cos_stride)
        new_k0 = k0 * cos_val - k1 * sin_val
        new_k1 = k1 * cos_val + k0 * sin_val

        split_idx_0 = i // x
        pos_in_split_0 = i % x
        split_idx_1 = (i + HEAD_DIM//2) // x
        pos_in_split_1 = (i + HEAD_DIM//2) % x

        k_cache_ptr_0 = (
            k_cache
            + block_number * kcb_stride
            + kv_head_idx * kch_stride
            + split_idx_0 * kcsplit_x_stride
            + offset_in_block * kcs_stride
            + pos_in_split_0 * kcd_stride
        )
        k_cache_ptr_1 = (
            k_cache
            + block_number * kcb_stride
            + kv_head_idx * kch_stride
            + split_idx_1 * kcsplit_x_stride
            + offset_in_block * kcs_stride
            + pos_in_split_1 * kcd_stride
        )
        tl.store(k_cache_ptr_0, new_k0)
        tl.store(k_cache_ptr_1, new_k1)

    # Update value cache
    for i in tl.static_range(0, HEAD_DIM // 2):
        v0 = tl.load(v_ptr + i * head_dim_stride)
        v1 = tl.load(v_ptr + (i + HEAD_DIM//2) * head_dim_stride)
        cos_val = tl.load(cos_ptr + i * cos_stride)
        sin_val = tl.load(sin_ptr + i * cos_stride)
        new_v0 = v0 * cos_val - v1 * sin_val
        new_v1 = v1 * cos_val + v0 * sin_val

        split_idx_0 = i // x
        pos_in_split_0 = i % x
        split_idx_1 = (i + HEAD_DIM//2) // x
        pos_in_split_1 = (i + HEAD_DIM//2) % x

        v_cache_ptr_0 = (
            v_cache
            + block_number * vcb_stride
            + kv_head_idx * vch_stride
            + split_idx_0 * vcs_stride
            + offset_in_block * vcs_stride
            + pos_in_split_0 * vcd_stride
        )
        v_cache_ptr_1 = (
            v_cache
            + block_number * vcb_stride
            + kv_head_idx * vch_stride
            + split_idx_1 * vcs_stride
            + offset_in_block * vcs_stride
            + pos_in_split_1 * vcd_stride
        )
        tl.store(v_cache_ptr_0, new_v0)
        tl.store(v_cache_ptr_1, new_v1)

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
    kcsplit_x_stride, kcs_stride, kcd_stride = 0, k_cache.stride(2), k_cache.stride(3) if k_cache is not None else (0,0,0)

    if use_new_kcache_layout:
        assert k_cache.dim() == 5, "KCache must be 5D with new layout"
        x = k_cache.size(-1)
        kcsplit_x_stride, kcs_stride, kcd_stride = k_cache.stride()[-3:]

    grid = (q_head_num, q_total_tokens)
    decoding_fused_rotary_embedding_kernel[grid](
        q, k, v, cos, sin, k_cache, v_cache, block_tables, kv_lengths, x,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), q.stride(2),
        cos.stride(0), cos.stride(1),
        k_cache.stride(0) if k_cache is not None else 0,
        k_cache.stride(1) if k_cache is not None else 0,
        kcsplit_x_stride, kcs_stride, kcd_stride,
        v_cache.stride(0) if v_cache is not None else 0,
        v_cache.stride(1) if v_cache is not None else 0,
        v_cache.stride(2) if v_cache is not None else 0,
        v_cache.stride(3) if v_cache is not None else 0,
        block_tables.stride(0) if block_tables is not None else 0,
        block_tables.stride(1) if block_tables is not None else 0,
        k_cache.size(-2) if k_cache is not None else 0,
        KV_GROUP_NUM=q_head_num // k_head_num,
        HEAD_DIM=head_dim,
        num_warps=16 if head_dim >= 512 else 8 if head_dim >= 256 else 4,
    )
    return q
