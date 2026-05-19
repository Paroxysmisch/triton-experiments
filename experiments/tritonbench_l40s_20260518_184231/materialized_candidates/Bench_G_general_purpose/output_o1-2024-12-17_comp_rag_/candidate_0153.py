import torch
import triton
import triton.language as tl
import warnings
from typing import Optional

@triton.jit
def rotary_embedding_kernel(
    q,
    k,
    cos,
    sin,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # calculate offsets
    head_idx = pid_m
    token_idx = pid_n * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    token_mask = token_idx < q_total_tokens

    # base pointers
    q_ptrs = q + head_idx * q_head_stride + token_idx * q_token_stride
    k_ptrs = k + head_idx * k_head_stride + token_idx * k_token_stride
    cos_ptrs = cos + token_idx * cos_token_stride
    sin_ptrs = sin + token_idx * cos_token_stride

    # HEAD_DIM is divided into two halves for rotary
    # compute rotation
    offs = tl.arange(0, HEAD_DIM)
    half_size = HEAD_DIM // 2
    half_mask = offs < half_size

    # gather q/k
    q_data = tl.load(q_ptrs + offs, mask=token_mask & (offs < HEAD_DIM), other=0.0)
    k_data = tl.load(k_ptrs + offs, mask=token_mask & (offs < HEAD_DIM), other=0.0)
    cos_val = tl.load(cos_ptrs + offs, mask=token_mask & (offs < HEAD_DIM), other=0.0)
    sin_val = tl.load(sin_ptrs + offs, mask=token_mask & (offs < HEAD_DIM), other=0.0)

    # apply rotary embedding
    # separate into two halves
    q_heada = tl.where(half_mask, q_data, 0.0)
    q_headb = tl.where(half_mask, 0.0, q_data)
    k_heada = tl.where(half_mask, k_data, 0.0)
    k_headb = tl.where(half_mask, 0.0, k_data)

    # rotate
    q_rotated = q_heada * cos_val - q_headb * sin_val + \
                q_headb * cos_val + q_heada * sin_val
    k_rotated = k_heada * cos_val - k_headb * sin_val + \
                k_headb * cos_val + k_heada * sin_val

    # store result back
    tl.store(q_ptrs + offs, q_rotated, mask=token_mask & (offs < HEAD_DIM))
    tl.store(k_ptrs + offs, k_rotated, mask=token_mask & (offs < HEAD_DIM))

@triton.jit
def fused_rotary_embedding_kernel_v2(
    q,
    k,
    cos,
    sin,
    kv_cache,
    BLOCK_TABLES,
    context_lengths,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    cacheb_stride,
    cacheh_stride,
    cachebs_stride,
    cached_stride,
    bts_stride,
    btb_stride,
    block_size,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    head_idx = pid_m
    token_idx = pid_n

    # load q/k for the token and head
    q_ptrs = q + head_idx * q_head_stride + token_idx * q_token_stride
    k_ptrs = k + head_idx * k_head_stride + token_idx * k_token_stride
    cos_ptrs = cos + token_idx * cos_token_stride
    sin_ptrs = sin + token_idx * cos_token_stride

    offs = tl.arange(0, HEAD_DIM)
    mask = offs < HEAD_DIM

    q_data = tl.load(q_ptrs + offs, mask=mask, other=0.0)
    k_data = tl.load(k_ptrs + offs, mask=mask, other=0.0)
    cos_val = tl.load(cos_ptrs + offs, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptrs + offs, mask=mask, other=0.0)

    half_size = HEAD_DIM // 2
    half_mask = offs < half_size

    q_heada = tl.where(half_mask, q_data, 0.0)
    q_headb = tl.where(half_mask, 0.0, q_data)
    k_heada = tl.where(half_mask, k_data, 0.0)
    k_headb = tl.where(half_mask, 0.0, k_data)

    q_rotated = q_heada * cos_val - q_headb * sin_val + \
                q_headb * cos_val + q_heada * sin_val
    k_rotated = k_heada * cos_val - k_headb * sin_val + \
                k_headb * cos_val + k_heada * sin_val

    tl.store(q_ptrs + offs, q_rotated, mask=mask)
    tl.store(k_ptrs + offs, k_rotated, mask=mask)

    # update kv_cache
    # simplistic example for demonstration
    cb_ptrs = kv_cache + head_idx * cacheh_stride
    c_ptrs = cb_ptrs + token_idx * cacheb_stride
    c_offs = c_ptrs + offs

    # store k_rotated in kv_cache
    tl.store(c_offs, k_rotated, mask=mask)

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
    pid_m = tl.program_id(0)  # head
    pid_n = tl.program_id(1)  # token

    head_idx = pid_m
    token_idx = pid_n

    # pointers to q, k, v
    q_ptrs = q + head_idx * q_head_stride + token_idx * q_token_stride
    k_ptrs = k + head_idx * k_head_stride + token_idx * k_token_stride
    v_ptrs = v + head_idx * k_head_stride + token_idx * k_token_stride

    # pointers to cos, sin
    cos_ptrs = cos + token_idx * cos_token_stride
    sin_ptrs = sin + token_idx * cos_token_stride

    offs = tl.arange(0, HEAD_DIM)
    mask = offs < HEAD_DIM

    # load q, k, v
    q_data = tl.load(q_ptrs + offs, mask=mask, other=0.0)
    k_data = tl.load(k_ptrs + offs, mask=mask, other=0.0)
    v_data = tl.load(v_ptrs + offs, mask=mask, other=0.0)
    cos_val = tl.load(cos_ptrs + offs, mask=mask, other=0.0)
    sin_val = tl.load(sin_ptrs + offs, mask=mask, other=0.0)

    half_size = HEAD_DIM // 2
    half_mask = offs < half_size

    # rotary for q and k
    q_heada = tl.where(half_mask, q_data, 0.0)
    q_headb = tl.where(half_mask, 0.0, q_data)
    k_heada = tl.where(half_mask, k_data, 0.0)
    k_headb = tl.where(half_mask, 0.0, k_data)

    q_rotated = q_heada * cos_val - q_headb * sin_val + \
                q_headb * cos_val + q_heada * sin_val
    k_rotated = k_heada * cos_val - k_headb * sin_val + \
                k_headb * cos_val + k_heada * sin_val

    # store back to q
    tl.store(q_ptrs + offs, q_rotated, mask=mask)

    # update k_cache and v_cache
    # We interpret shapes/strides. This is a simplified example.
    # kcb_stride => batch
    # kch_stride => head
    # kcs_stride => sequence
    # kcd_stride => dimension
    # optionally kcsplit_x_stride if new layout is used
    k_cache_ptrs = k_cache + head_idx * kch_stride + token_idx * kcb_stride
    v_cache_ptrs = v_cache + head_idx * vch_stride + token_idx * vcb_stride
    k_cache_offs = k_cache_ptrs + offs
    v_cache_offs = v_cache_ptrs + offs

    # store k_rotated
    tl.store(k_cache_offs, k_rotated, mask=mask)
    # store v as is
    tl.store(v_cache_offs, v_data, mask=mask)

@triton.jit
def rotary_embedding(
    q: tl.tensor,
    k: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    k_cache: tl.tensor,
    block_tables: tl.tensor,
    kv_lengths: tl.tensor,
):
    # This is just a placeholder to illustrate a possible signature
    pass

def rotary_embedding_host(
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
            q,
            k,
            cos,
            sin,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            q.stride(2),
            cos.stride(0),
            cos.stride(1),
            q_total_tokens,
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
            q,
            k,
            cos,
            sin,
            k_cache,
            block_tables,
            kv_lengths,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            q.stride(2),
            cos.stride(0),
            cos.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            k_cache.size(-2),
            q_total_tokens,
            Q_HEAD_NUM=q_head_num,
            HEAD_DIM=head_dim,
            num_warps=16 if head_dim >= 512 else 8 if head_dim >= 256 else 4,
        )
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
