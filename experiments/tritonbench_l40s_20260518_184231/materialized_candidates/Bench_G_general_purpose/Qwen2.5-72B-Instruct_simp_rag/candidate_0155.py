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
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    head_id = pid
    token_id = bid

    # Calculate the start and end indices for the block
    start = token_id * block_size
    end = start + block_size

    # Load the query, key, and value for the current head and token block
    q_ptr = q + head_id * q_head_stride + start * q_token_stride
    k_ptr = k + head_id * k_head_stride + start * k_token_stride
    v_ptr = v + head_id * q_head_stride + start * q_token_stride

    q_block = tl.load(q_ptr, mask=start + tl.arange(0, block_size) < q.size(0), other=0.0)
    k_block = tl.load(k_ptr, mask=start + tl.arange(0, block_size) < k.size(0), other=0.0)
    v_block = tl.load(v_ptr, mask=start + tl.arange(0, block_size) < v.size(0), other=0.0)

    # Load the cosine and sine values for the current token block
    cos_ptr = cos + start * cos_token_stride
    sin_ptr = sin + start * cos_token_stride

    cos_block = tl.load(cos_ptr, mask=start + tl.arange(0, block_size) < cos.size(0), other=0.0)
    sin_block = tl.load(sin_ptr, mask=start + tl.arange(0, block_size) < sin.size(0), other=0.0)

    # Apply rotary embedding to the query and key
    q_rotated = q_block * cos_block - q_block * sin_block
    k_rotated = k_block * cos_block + k_block * sin_block

    # Store the rotated query and key back to the original tensors
    tl.store(q_ptr, q_rotated, mask=start + tl.arange(0, block_size) < q.size(0))
    tl.store(k_ptr, k_rotated, mask=start + tl.arange(0, block_size) < k.size(0))

    # Update the key and value caches
    k_cache_ptr = k_cache + head_id * kch_stride + start * kcd_stride
    v_cache_ptr = v_cache + head_id * vch_stride + start * vcd_stride

    tl.store(k_cache_ptr, k_rotated, mask=start + tl.arange(0, block_size) < k.size(0))
    tl.store(v_cache_ptr, v_block, mask=start + tl.arange(0, block_size) < v.size(0))

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
