import torch
import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr,
    k_cache_ptr, v_cache_ptr,
    BLOCK_TABLES, context_lengths,
    x,
    q_token_stride, q_head_stride, k_token_stride, k_head_stride, head_dim_stride,
    cos_token_stride, cos_stride,
    kcb_stride, kch_stride, kcsplit_x_stride, kcs_stride, kcd_stride,
    vcb_stride, vch_stride, vcs_stride, vcd_stride,
    bts_stride, btb_stride,
    block_size,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)

    # Load the necessary data
    q = tl.load(q_ptr + pid * q_token_stride + bid * q_head_stride)
    k = tl.load(k_ptr + pid * k_token_stride + bid * k_head_stride)
    v = tl.load(v_ptr + pid * q_token_stride + bid * q_head_stride)

    cos = tl.load(cos_ptr + pid * cos_token_stride + bid * cos_stride)
    sin = tl.load(sin_ptr + pid * cos_token_stride + bid * cos_stride)

    # Apply rotary embedding transformation
    q_rot = q * cos - tl.swizzle(q, 'xor') * sin
    k_rot = k * cos - tl.swizzle(k, 'xor') * sin

    # Store the transformed q and k back
    tl.store(q_ptr + pid * q_token_stride + bid * q_head_stride, q_rot)
    tl.store(k_ptr + pid * k_token_stride + bid * k_head_stride, k_rot)

    # Update caches
    tl.store(k_cache_ptr + pid * kcb_stride + bid * kch_stride, k_rot)
    tl.store(v_cache_ptr + pid * vcb_stride + bid * vch_stride, v)

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
