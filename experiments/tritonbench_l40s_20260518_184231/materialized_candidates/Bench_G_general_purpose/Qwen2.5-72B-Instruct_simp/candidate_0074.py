import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K_ptr, V_ptr, KCache_ptr, VCache_ptr,
    bsz, num_kv_heads, head_dim, cache_head_dim,
    KCache_stride_b, KCache_stride_h, KCache_stride_d,
    VCache_stride_b, VCache_stride_h, VCache_stride_d,
    K_stride_b, K_stride_h, K_stride_d,
    V_stride_b, V_stride_h, V_stride_d,
    cache_offset,
    BLOCK_HEAD: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // num_kv_heads
    hid = pid % num_kv_heads

    # Compute the base pointers for K and V
    K_base = K_ptr + bid * K_stride_b + hid * K_stride_h
    V_base = V_ptr + bid * V_stride_b + hid * V_stride_h

    # Compute the base pointers for KCache and VCache
    KCache_base = KCache_ptr + bid * KCache_stride_b + hid * KCache_stride_h + cache_offset * KCache_stride_d
    VCache_base = VCache_ptr + bid * VCache_stride_b + hid * VCache_stride_h + cache_offset * VCache_stride_d

    # Iterate over the head dimension in blocks
    for d in range(0, head_dim, BLOCK_D):
        K_offset = d * K_stride_d
        V_offset = d * V_stride_d
        KCache_offset = d * KCache_stride_d
        VCache_offset = d * VCache_stride_d

        K_block_ptr = K_base + K_offset
        V_block_ptr = V_base + V_offset
        KCache_block_ptr = KCache_base + KCache_offset
        VCache_block_ptr = VCache_base + VCache_offset

        for i in range(BLOCK_D):
            if d + i < head_dim:
                K_val = tl.load(K_block_ptr + i * K_stride_d)
                V_val = tl.load(V_block_ptr + i * V_stride_d)
                tl.store(KCache_block_ptr + i * KCache_stride_d, K_val)
                tl.store(VCache_block_ptr + i * VCache_stride_d, V_val)

import torch

def copy_kv_to_blocked_cache(K, V, KCache, VCache, cache_offset, bsz, num_kv_heads, head_dim, cache_head_dim):
    assert K.shape == (bsz, num_kv_heads, head_dim), "K tensor shape mismatch"
    assert V.shape == (bsz, num_kv_heads, head_dim), "V tensor shape mismatch"
    assert KCache.shape == (bsz, num_kv_heads, cache_head_dim), "KCache tensor shape mismatch"
    assert VCache.shape == (bsz, num_kv_heads, cache_head_dim), "VCache tensor shape mismatch"

    # Define grid and block dimensions
    grid = (bsz * num_kv_heads, )
    block = (1, )

    # Launch the kernel
    _copy_to_kvcache_seqlen1_kernel[grid, block](
        K, V, KCache, VCache,
        bsz, num_kv_heads, head_dim, cache_head_dim,
        KCache.stride(0), KCache.stride(1), KCache.stride(2),
        VCache.stride(0), VCache.stride(1), VCache.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        cache_offset,
        BLOCK_HEAD=1, BLOCK_D=32  # Adjust BLOCK_D as needed
    )

import torch

# Example tensors
bsz = 2
num_kv_heads = 4
head_dim = 64
cache_head_dim = 128
cache_offset = 0

K = torch.randn(bsz, num_kv_heads, head_dim, device='cuda')
V = torch.randn(bsz, num_kv_heads, head_dim, device='cuda')
KCache = torch.zeros(bsz, num_kv_heads, cache_head_dim, device='cuda')
VCache = torch.zeros(bsz, num_kv_heads, cache_head_dim, device='cuda')

# Call the wrapper function
copy_kv_to_blocked_cache(K, V, KCache, VCache, cache_offset, bsz, num_kv_heads, head_dim, cache_head_dim)
