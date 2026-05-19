import torch
import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scales_ptr,
    zeros_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scales_g,
    stride_scales_n,
    stride_zeros_g,
    stride_zeros_n,
    groupsize: tl.constexpr,
    NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Computes: C = A * B
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (NGROUPS, N) float16
    zeros is of shape (NGROUPS, N//8) int32
    """

    infearure_per_bits = 32 // 4

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_m = pid // (num_pid_n * num_pid_k)
    pid_n = (pid % (num_pid_n * num_pid_k)) // num_pid_k
    pid_k = (pid % (num_pid_n * num_pid_k)) % num_pid_k
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (
        (offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn
    )
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    shifter = (offs_k % infearure_per_bits) * 4

    NGROUPS = (K + groupsize - 1) // groupsize

    group_id = offs_k[0] // groupsize
    group_size = min(K - groupsize * group_id, groupsize)
    scales_ptrs = scales_ptr + stride_scales_g * group_id + stride_scales_n * offs_bn[None, :]
    zeros_ptrs = zeros_ptr + stride_zeros_g * group_id + stride_zeros_n * (offs_bn[None, :] // infearure_per_bits)
    zeros_shifter = (offs_bn % infearure_per_bits) * 4

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        zeros = tl.load(zeros_ptrs)
        zeros = (zeros >> zeros_shifter[None, :]) & 0xF
        zeros = zeros + 1

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
        b = (b - zeros) * tl.load(scales_ptrs)  # Unpack and dequantize
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
        scales_ptrs += BLOCK_SIZE_N
        zeros_ptrs += (BLOCK_SIZE_N // infearure_per_bits) * stride_zeros_n

    c = accumulator.to(tl.float16)
    tl.store(c_ptrs, c)  # Note: Uses non-unit strides to stay within bounds

@triton.jit
def dequantize_kernel(
    b_ptr,
    b_scale_ptr,
    b_zp_ptr,
    fpb_ptr,
    K,
    N,
    group_size,
    stride_bk,
    stride_bn,
    stride_bsk,
    stride_bsn,
    stride_bzpk,
    stride_bzpn,
    stride_fpbk,
    stride_fpbn,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Dequantizes int4 weights into float16.
    """

    pid = tl.program_id(axis=0)

    NGROUPS = (K + group_size - 1) // group_size

    # We need to figure out the group bounds. This is done by considering
    # the program ID and the group size.
    group_id = pid
    grid_size = triton.cdiv(K, BLOCK_SIZE_K)
    num_full_groups = (grid_size - 1) // NGROUPS
    num_remaining_weights = grid_size - num_full_groups * NGROUPS
    if group_id >= num_full_groups:
        group_size = group_size - (num_full_groups * NGROUPS - group_id) * group_size
    # Each program is responsible for a different group, and we compute the
    # the the first group in the grid.
    group_offset_k = group_id * group_size

    offs_k = group_offset_k + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    b_ptrs = b_ptr + offs_k * stride_bk + offs_n * stride_bn
    fpb_ptrs = fpb_ptr + offs_k * stride_fpbk + offs_n * stride_fpbn

    # Each block of pids 0,..,N-1 corrsponds to one group.
    # Compute the float value.
    b_tile = tl.load(b_ptrs)

    # Compute the zero point
    b_zero_point_tile = tl.load(b_zp_ptr)
    # still buggy for strided
    # b_zero_point_tile = tl.load(b_zp_ptr + group_id*stride_bzpk + offs_n*stride_bzpn)
    b_scale_tile = tl.load(b_scale_ptr)
    # still buggy for strided
    # b_scale_tile = tl.load(b_scale_ptr + offs_n*stride_bsn)

    fp_weight = (b_tile - b_zero_point_tile) * b_scale_tile
    tl.store(fpb_ptrs, fp_weight
