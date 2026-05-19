import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Reorder program IDs to improve L2 cache behavior
    width = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    group_id = pid_m // GROUP_SIZE_M
    group_size = GROUP_SIZE_M * width
    pid_m = group_id * GROUP_SIZE_M + (pid_m % GROUP_SIZE_M)

    # Compute the tile coordinates
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create a mask for in-bounds access
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator in int32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    # Loop over K dimension in steps of (BLOCK_SIZE_K)
    # Each step in K dimension processes 4 int8 values from 'a' and 1 uint8 value from 'b'
    # repeated (BLOCK_SIZE_K // 1) times, with partial unrolling of 4 int8 elements.
    for k_tile in range(0, K, BLOCK_SIZE_K):
        # Offsets for 'a' and 'b'
        a_offs = (offs_m[:, None] * stride_am) + ((k_tile * 4) * stride_ak)
        b_offs = ((k_tile) * stride_bk) + (offs_n[None, :] * stride_bn)

        # Accumulate in steps of 4 int8 from 'a' vs. 1 uint8 from 'b'
        # This partial unrolling processes 4 columns from 'a' for each iteration in 'k_tile'
        for kk in range(BLOCK_SIZE_K):
            # Load 4 int8 elements from 'a' as one int32
            # Each row gets one int32 containing 4 int8
            a_val_int32 = tl.load(a_ptr + a_offs + kk * 4 * stride_ak, mask=mask_m[:, None], other=0, dtype=tl.int32)
            # Load one uint8 element from 'b' as int32
            b_val_int32 = tl.load(b_ptr + b_offs + kk * stride_bk, mask=mask_n[None, :], other=0, dtype=tl.int32)

            # Extract 4 int8 from a_val_int32 using shifting and masking
            a0 = ((a_val_int32 >> 0)  & 0xff).to(tl.int8)
            a1 = ((a_val_int32 >> 8)  & 0xff).to(tl.int8)
            a2 = ((a_val_int32 >> 16) & 0xff).to(tl.int8)
            a3 = ((a_val_int32 >> 24) & 0xff).to(tl.int8)

            # Sign-extend each int8
            a0 = a0.astype(tl.int32)
            a1 = a1.astype(tl.int32)
            a2 = a2.astype(tl.int32)
            a3 = a3.astype(tl.int32)

            # b_val_int32 is uint8 per element, but loaded as int32
            # mask to avoid sign extension
            b_val_int32 &= 0xff

            # Accumulate partial products
            # (a0..a3) broadcast along columns, b_val_int32 broadcast along rows
            # shift creates effectively 4 accumulations
            acc += (a0 * b_val_int32)
            acc += (a1 * b_val_int32)
            acc += (a2 * b_val_int32)
            acc += (a3 * b_val_int32)

        # Advance 'a_offs' and 'b_offs' are automatically adjusted in the loop

    # Store results back to c
    c_offs = (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptr + c_offs, acc, mask=mask_m[:, None] & mask_n[None, :])


def matmul(a, b):
    import math
    # a: (M, 4*K) int8
    # b: (K, N) uint
