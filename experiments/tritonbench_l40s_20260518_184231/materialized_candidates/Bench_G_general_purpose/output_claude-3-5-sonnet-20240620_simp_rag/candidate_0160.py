import torch
import triton
import triton.language as tl
import warnings
from typing import Optional

@triton.jit
def rotary_embedding_kernel(
    q, k, cos, sin,
    q_token_stride, q_head_stride, k_token_stride, k_head_stride,
    head_dim_stride, cos_token_stride, cos_stride, q_total_tokens,
    Q_HEAD_NUM: tl.constexpr, KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_TOKENS: tl.constexpr,
):
    # Compute program ID
    pid = tl.program_id(0)
    bid = tl.program_id(1)

    # Compute offsets
    head_id = pid
    token_id = bid * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)

    # Compute pointers
    q_ptr = q + head_id * q_head_stride + token_id[:, None] * q_token_stride + tl.arange(0, HEAD_DIM)[None, :] * head_dim_stride
    k_ptr = k + (head_id // KV_GROUP_NUM) * k_head_stride + token_id[:, None] * k_token_stride + tl.arange(0, HEAD_DIM)[None, :] * head_dim_stride
    cos_ptr = cos + token_id[:, None] * cos_token_stride + tl.arange(0, HEAD_DIM)[None, :] * cos_stride
    sin_ptr = sin + token_id[:, None] * cos_token_stride + tl.arange(0, HEAD_DIM)[None, :] * cos_stride

    # Load data
    q_val = tl.load(q_ptr, mask=token_id[:, None] < q_total_tokens, other=0.0)
    k_val = tl.load(k_ptr, mask=token_id[:, None] < q_total_tokens, other=0.0)
    cos_val = tl.load(cos_ptr, mask=token_id[:, None] < q_total_tokens, other=1.0)
    sin_val = tl.load(sin_ptr, mask=token_id[:, None] < q_total_tokens, other=0.0)

    # Apply rotary embedding
    q_rot = q_val * cos_val + tl.where(tl.arange(0, HEAD_DIM)[None, :] % 2 == 0,
                                       -q_val[:, 1:] * sin_val[:, :-1],
                                       q_val[:, :-1] * sin_val[:, 1:])
    k_rot = k_val * cos_val + tl.where(tl.arange(0, HEAD_DIM)[None, :] % 2 == 0,
                                       -k_val[:, 1:] * sin_val[:, :-1],
                                       k_val[:, :-1] * sin_val[:, 1:])

    # Store results
    tl.store(q_ptr, q_rot, mask=token_id[:, None] < q_total_tokens)
    tl.store(k_ptr, k_rot, mask=token_id[:, None] < q_total_tokens)

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q, k, v, cos, sin, k_cache, v_cache, BLOCK_TABLES, context_lengths, x,
    q_token_stride, q_head_stride, k_token_stride, k_head_stride,
    head_dim_stride, cos_token_stride, cos_stride,
    kcb_stride, kch_stride, kcsplit_x_stride, kcs_stride, kcd_stride,
    vcb_stride, vch_stride, vcs_stride, vcd_stride,
    bts_stride, btb_stride, block_size,
    KV_GROUP_NUM: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # Compute program ID
    pid = tl.program_id(0)
    bid = tl.program_id(1)

    # Compute offsets
    head_id = pid
    token_id = bid

    # Compute pointers
    q_ptr = q + head_id * q_head_stride + token_id * q_token_stride + tl.arange(0, HEAD_DIM) * head_dim_stride
    k_ptr = k + (head_id // KV_GROUP_NUM) * k_head_stride + token_id * k_token_stride + tl.arange(0, HEAD_DIM) * head_dim_stride
    v_ptr = v + (head_id // KV_GROUP_NUM) * k_head_stride + token_id * k_token_stride + tl.arange(0, HEAD_DIM) * head_dim_stride
    cos_ptr = cos + token_id * cos_token_stride + tl.arange(0, HEAD_DIM) * cos_stride
    sin_ptr = sin + token_id * cos_token_stride + tl.arange(0, HEAD_DIM) * cos_stride

    # Load data
    q_val = tl.load(q_ptr)
    k_val = tl.load(k_ptr)
    v_val = tl.load(v_ptr)
    cos_val = tl.load(cos_ptr)
    sin_val = tl.load(sin_ptr)

    # Apply rotary embedding
    q_rot = q_val * cos_val + tl.where(tl.arange(0, HEAD_DIM) % 2 == 0,
                                       -q_val[1:] * sin_val[:-1],
                                       q_val[:-1] * sin_val[1:])
    k_rot = k_val * cos_val + tl.where(tl.arange(0, HEAD_DIM) % 2 == 0,
                                       -k_val[1:] * sin_val[:-1],
                                       k_val[:-1] * sin_val[1:])

    # Store results for q, k, v
    tl.store(q_ptr, q_rot)
    tl.store(k_ptr, k_rot)
    tl.store(v_ptr, v_val)

    # Update k_cache and v_cache
    batch_id = token_id // block_size
    block_table = BLOCK_TABLES + batch_id * bts_stride
    block_id = tl.load(block_table + (token_id % block_size) * btb_stride)
    context_len = tl.load(context_lengths + batch_id)

    k_cache_ptr = k_cache + block_id * kcb_stride + (head_id // KV_GROUP_NUM) * kch_stride + \
                  (token_id % block_size) * kcs_stride + tl.arange(0, HEAD_DIM) * kcd_stride
    v_cache_ptr = v_cache + block_id * vcb_stride + (head_id // KV_GROUP_NUM) * vch_stride + \
                  (token_id % block_size) * vcs_stride + tl.arange(0, HEAD_DIM) * vcd_stride

    tl.store(k_cache_ptr, k_rot)
    tl.store(v_cache_ptr, v_val)

def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
):
    q_total_tokens, q_head_num, head_dim = q.shape

    if k_cache is None:
        grid = lambda META: (
            q_head_num,
            triton.cdiv(q_total_tokens, META["BLOCK_TOKENS"]),
        )
        rotary_embedding_kernel[grid](
            q, k, cos, sin,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            q.stride(2), cos.stride(0), cos.stride(1), q_total_tokens,
            Q_HEAD_NUM=q_head_num,
            KV_GROUP_NUM=q_head_num // k.size(1),
            HEAD_DIM=head_dim,
            BLOCK_TOKENS=4,
            num_warps=16 if head_dim >= 512 else 8 if head_dim >= 256 else 4,
        )
    else:
        warnings.warn("Fused rotary embedding Triton kernel will be deprecated as the new kcache layout is supported")
        grid = (triton.next_power_of_2(q_head_num), q_total_tokens)
        fused_rotary_embedding_kernel_v2[grid](
            q, k, cos, sin, k_cache, block_tables, kv_lengths,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), q.stride(2),
            cos.stride(0), cos.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            block_tables.stride(0), block_tables.stride(1),
            k_cache.size(-2), q_total_tokens,
            Q_HEAD_NUM=q_head_num,
            HEAD_DIM=head_dim,
            num_warps=16 if head_dim >= 512 else 8 if head_dim >= 256 else 4,
        )

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
        q, k, v, cos, sin, k_cache, v_cache, block_tables, kv_lengths, x,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), q.stride(2),
        cos.stride(0), cos.stride(1),
        k_cache.stride(0), k_cache.stride(1), kcsplit_x_stride, kcs_stride, kcd_stride,
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        k_cache.size(-2),
        KV_GROUP_NUM=q_head_num // k_head_num,
        HEAD_DIM=head_dim,
        num_warps=16 if head_dim >= 512 else 8 if head_dim >= 256 else 4,
    )
