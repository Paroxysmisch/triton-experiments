import triton
import triton.language as tl
import torch


@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K_ptr, V_ptr, KCache_ptr, VCache_ptr,
    block_tables_ptr, context_lengths_ptr,
    stride_k_b, stride_k_h, stride_k_d,
    stride_v_b, stride_v_h, stride_v_d,
    stride_kc_g, stride_kc_b, stride_kc_h, stride_kc_s, stride_kc_d,
    stride_vc_g, stride_vc_b, stride_vc_h, stride_vc_s, stride_vc_d,
    batch_size, num_heads, block_size, head_dim,
    use_new_cache_layout,
    BLOCK_DIM: tl.constexpr
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    seq_offset = tl.load(context_lengths_ptr + b_idx)
    block_idx = tl.load(block_tables_ptr + b_idx * stride_k_b + seq_offset)
    offset_in_block = seq_offset % block_size

    dim_range = tl.arange(0, BLOCK_DIM)
    mask = dim_range < head_dim

    k_vals = tl.load(
        K_ptr + b_idx * stride_k_b + h_idx * stride_k_h + dim_range * stride_k_d,
        mask=mask,
        other=0.0
    )
    v_vals = tl.load(
        V_ptr + b_idx * stride_v_b + h_idx * stride_v_h + dim_range * stride_v_d,
        mask=mask,
        other=0.0
    )

    if use_new_cache_layout:
        # new layout: [num_blocks, num_kv_heads, head_dim // X, block_size, X]
        # suppose the factor X is a compile-time known factor, e.g. 32
        # split dim_range into major/minor for the last dimension
        # here we assume X == BLOCK_DIM to illustrate
        # (Adjust indexing for general factor X accordingly.)
        minor_range = dim_range % BLOCK_DIM
        major_range = dim_range // BLOCK_DIM

        k_offset = (
            block_idx * stride_kc_g
            + h_idx * stride_kc_h
            + major_range * stride_kc_s
            + offset_in_block * stride_kc_d
            + minor_range
        )
        v_offset = (
            block_idx * stride_vc_g
            + h_idx * stride_vc_h
            + major_range * stride_vc_s
            + offset_in_block * stride_vc_d
            + minor_range
        )
    else:
        # old layout: [num_blocks, num_kv_heads, block_size, head_dim]
        k_offset = (
            block_idx * stride_kc_g
            + h_idx * stride_kc_h
            + offset_in_block * stride_kc_s
            + dim_range
        )
        v_offset = (
            block_idx * stride_vc_g
            + h_idx * stride_vc_h
            + offset_in_block * stride_vc_s
            + dim_range
        )

    tl.store(KCache_ptr + k_offset, k_vals, mask=mask)
    tl.store(VCache_ptr + v_offset, v_vals, mask=mask)


def copy_kv_to_blocked_cache(
    K,
    V,
    KCache,
    VCache,
    block_tables,
    context_lengths,
    use_new_cache_layout: bool
):
    # K, V shapes: [batch_size, num_heads, head_dim]
    # KCache, VCache shapes can be one of:
    #   old: [num_blocks, num_kv_heads, block_size, head_dim]
    #   new: [num_blocks, num_kv_heads, head_dim // x, block_size, x]
    batch_size, num_heads, head_dim = K.shape
    block_size = KCache.shape[2] if not use_new_cache_layout else KCache.shape[3]

    # Strides for K, V
    stride_k_b = K.stride(0)
    stride_k_h = K.stride(1)
    stride_k_d = K.stride(2)
    stride_v_b = V.stride(0)
    stride_v_h = V.stride(1)
    stride_v_d = V.stride(2)

    # Strides for KCache, VCache
    if not use_new_cache_layout:
        # old layout
        stride_kc_g = KCache.stride(0)
        stride_kc_b = KCache.stride(1)
        stride_kc_s = KCache.stride(2)
        stride_kc_d = KCache.stride(3)
        stride_vc_g = VCache.stride(0)
        stride_vc_b = VCache.stride(1)
        stride_vc_s = VCache.stride(2)
        stride_vc_d = VCache.stride(3)
    else:
        # new layout
        stride_kc_g = KCache.stride(0)
        stride_kc_b = KCache.stride(1)
        stride_kc_s = KCache.stride(2)
        stride_kc_d = KCache.stride(4)
        stride_vc_g = VCache.stride(0)
        stride_vc_b = VCache.stride(1)
        stride_vc_s = VCache.stride(2)
        stride_vc_d = VCache.stride(4)

    grid = (batch_size, num_heads)
    BLOCK_DIM = 128  # example block size for head_dim dimension

    triton.run(
        _copy_to_kvcache_seqlen1_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        args=[
            K,
            V,
            KCache,
            VCache,
            block_tables,
            context_lengths,
            stride_k_b,
            stride_k_h,
            stride_k_d,
            stride_v_b,
            stride_v_h,
            stride_v_d,
            stride_kc_g,
            stride_kc_b,
            stride_kc_h,
            stride_kc_s,
            stride_kc_d,
            stride_vc_g,
            stride_vc_b,
            stride_vc_h,
            stride_vc_s,
            stride_vc_d,
            batch_size,
            num_heads,
            block_size,
            head_dim,
            use_new_cache_layout
        ],
        kwargs={"BLOCK_DIM": BLOCK_DIM},
    )
