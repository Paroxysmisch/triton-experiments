import torch
import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q,
    k,
    v,
    cos,
    sin,
    k_cache,
    v_cache,
    block_tables,
    kv_lengths,
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
    pid = tl.program_id(axis=0)
    q_head_id = pid // q_total_tokens
    q_token_id = pid % q_total_tokens

    q_head_base = q_head_id * q_head_stride
    q_token_base = q_token_id * q_token_stride
    k_head_base = (q_head_id // KV_GROUP_NUM) * k_head_stride
    k_token_base = q_token_id * k_token_stride

    # Load q, k, v
    q_ptr = q + q_head_base + q_token_base
    k_ptr = k + k_head_base + k_token_base
    v_ptr = v + q_head_base + q_token_base

    q_val = tl.load(q_ptr, mask=q_token_id < q_total_tokens, other=0.0)
    k_val = tl.load(k_ptr, mask=q_token_id < q_total_tokens, other=0.0)
    v_val = tl.load(v_ptr, mask=q_token_id < q_total_tokens, other=0.0)

    # Load cos and sin
    cos_ptr = cos + q_token_id * cos_token_stride
    sin_ptr = sin + q_token_id * cos_token_stride
    cos_val = tl.load(cos_ptr, mask=q_token_id < q_total_tokens, other=1.0)
    sin_val = tl.load(sin_ptr, mask=q_token_id < q_total_tokens, other=0.0)

    # Apply rotary embedding
    q_rotated = q_val * cos_val + tl.flip(q_val, 1) * sin_val
    k_rotated = k_val * cos_val + tl.flip(k_val, 1) * sin_val

    # Store back to q
    tl.store(q_ptr, q_rotated, mask=q_token_id < q_total_tokens)

    # Update k_cache and v_cache
    if k_cache is not None and v_cache is not None:
        k_cache_ptr = k_cache + q_head_id * kch_stride + q_token_id * kcb_stride
        v_cache_ptr = v_cache + q_head_id * vch_stride + q_token_id * vcb_stride

        k_cache_val = tl.load(k_cache_ptr, mask=q_token_id < q_total_tokens, other=0.0)
        v_cache_val = tl.load(v_cache_ptr, mask=q_token_id < q_total_tokens, other=0.0)

        tl.store(k_cache_ptr, k_rotated, mask=q_token_id < q_total_tokens)
        tl.store(v_cache_ptr, v_val, mask=q_token_id < q_total_tokens)

    return

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
        q,
        k,
        v,
        cos,
        sin,
        k_cache,
        v_cache,
        block_tables,
        kv_lengths,
        x,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        q.stride(2),
        cos.stride(0),
        cos.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        kcsplit_x_stride,
        kcs_stride,
        kcd_stride,
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        k_cache.size(-2),
        KV_GROUP_NUM=q_head_num // k_head_num,
        HEAD_DIM=head_dim,
        num_warps=16 if head_dim >= 512 else 8 if head_dim >= 256 else 4,
    )
    return
