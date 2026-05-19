import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K_ptr, V_ptr, KCache_ptr, VCache_ptr,
    bsz, num_kv_heads, head_dim,
    K_stride0, K_stride1, K_stride2,
    V_stride0, V_stride1, V_stride2,
    KCache_stride0, KCache_stride1, KCache_stride2,
    VCache_stride0, VCache_stride1, VCache_stride2,
    BLOCK_SIZE: tl.constexpr
):
    # Define program ID for grid
    pid0 = tl.program_id(0)  # batch index
    pid1 = tl.program_id(1)  # head index

    # Compute the offsets for K and V
    K_offset = pid0 * K_stride0 + pid1 * K_stride1
    V_offset = pid0 * V_stride0 + pid1 * V_stride1

    # Compute the offsets for KCache and VCache
    KCache_offset = pid0 * KCache_stride0 + pid1 * KCache_stride1
    VCache_offset = pid0 * VCache_stride0 + pid1 * VCache_stride1

    # Load K and V values
    K = tl.load(K_ptr + K_offset + tl.arange(0, BLOCK_SIZE))
    V = tl.load(V_ptr + V_offset + tl.arange(0, BLOCK_SIZE))

    # Store values into KCache and VCache
    tl.store(KCache_ptr + KCache_offset + tl.arange(0, BLOCK_SIZE), K)
    tl.store(VCache_ptr + VCache_offset + tl.arange(0, BLOCK_SIZE), V)


def copy_kv_to_blocked_cache(K, V, KCache, VCache, bsz, num_kv_heads, head_dim):
    # Define block size for the kernel
    BLOCK_SIZE = head_dim

    # Compute strides for input and cache tensors
    K_stride0, K_stride1, K_stride2 = K.stride()
    V_stride0, V_stride1, V_stride2 = V.stride()
    KCache_stride0, KCache_stride1, KCache_stride2 = KCache.stride()
    VCache_stride0, VCache_stride1, VCache_stride2 = VCache.stride()

    # Launch the Triton kernel
    grid = (bsz, num_kv_heads)
    _copy_to_kvcache_seqlen1_kernel[grid](
        K, V, KCache, VCache,
        bsz, num_kv_heads, head_dim,
        K_stride0, K_stride1, K_stride2,
        V_stride0, V_stride1, V_stride2,
        KCache_stride0, KCache_stride1, KCache_stride2,
        VCache_stride0, VCache_stride1, VCache_stride2,
        BLOCK_SIZE=BLOCK_SIZE
    )
