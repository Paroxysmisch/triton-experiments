import triton
import triton.language as tl
import math

# Utility function replicating a possible "kernel_utils.mul" behavior
# for elementwise multiplication. Adjust or remove as needed.
def elementwise_mul(x, y):
    return x * y

@triton.jit
def matmul_kernel_with_block_pointers(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr = False,
    GROUP_M: tl.constexpr = 1
):
    # -----------------------------------------------------------
    # Grouped program ID splitting for block-level tiling
    # -----------------------------------------------------------
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    group_size = GROUP_M * grid_n
    group_id = pid // group_size
    within_group_id = pid % group_size
    pid_m = group_id * GROUP_M + within_group_id // grid_n
    pid_n = within_group_id % grid_n

    # -----------------------------------------------------------
    # Block-level coordinate offsets
    # -----------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # -----------------------------------------------------------
    # Create pointers to A and B for block loading
    # -----------------------------------------------------------
    # We will iterate over K in steps of BLOCK_K:
    # block_k_range = range(0, K, BLOCK_K)
    # We'll incorporate the chunk offset in the loop
    # We'll do block-level pointer offsets
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # We accumulate partial products in steps of BLOCK_K
    # Use a single loop over the K dimension
    k_block_range = K if EVEN_K else tl.cdiv(K, BLOCK_K) * BLOCK_K
    for k_start in range(0, k_block_range, BLOCK_K):
        # Offsets in A and B for the current K-block
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + (offs_m[:, None] * stride_am) + ((k_start + offs_k[None, :]) * stride_ak)
        b_ptrs = B + ((k_start + offs_k[:, None]) * stride_bk) + (offs_n[None, :] * stride_bn)

        # Mask computation to guard against out-of-bounds K
        mask_k = (k_start + offs_k) < K
        a_block = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0)
        b_block = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0)

        # Integer dot-product accumulation
        accumulator += tl.dot(a_block.to(tl.int32), b_block.to(tl.int32))

    # Final elementwise multiplication: C = accumulator * accumulator
    c_block = accumulator * accumulator

    # Write results back
    c_ptrs = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    mask_store = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c_block, mask=mask_store)


@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    A, B, scales1, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_scale,  # stride for scales1 if it's a matrix
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr = False,
    GROUP_M: tl.constexpr = 1
):
    # -----------------------------------------------------------
    # Grouped program ID splitting for block-level tiling
    # -----------------------------------------------------------
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    group_size = GROUP_M * grid_n
    group_id = pid // group_size
    within_group_id = pid % group_size
    pid_m = group_id * GROUP_M + within_group_id // grid_n
    pid_n = within_group_id % grid_n

    # -----------------------------------------------------------
    # Block-level coordinate offsets
    # -----------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # -----------------------------------------------------------
    # Accumulator for integer matrix multiplication
    # -----------------------------------------------------------
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # -----------------------------------------------------------
    # Loop over K dimension
    # -----------------------------------------------------------
    k_block_range = K if EVEN_K else tl.cdiv(K, BLOCK_K) * BLOCK_K
    for k_start in range(0, k_block_range, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A + (offs_m[:, None] * stride_am) + ((k_start + offs_k[None, :]) * stride_ak)
        b_ptrs = B + ((k_start + offs_k[:, None]) * stride_bk) + (offs_n[None, :] * stride_bn)

        mask_k = (k_start + offs_k) < K
        a_block = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0)
        b_block = tl.load(b_ptrs, mask=mask_k[:, None
