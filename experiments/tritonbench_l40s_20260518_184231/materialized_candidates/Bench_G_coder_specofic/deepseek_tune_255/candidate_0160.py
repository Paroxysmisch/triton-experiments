import triton
import triton.language as tl
import torch
from torch import Tensor
from torch.utils._triton import instance_descriptor

@triton.jit
def matmul4_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    k_remainder: tl.constexpr, n_remainder: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
    USE_INITIAL_C: tl.constexpr,
    STORE_FINAL_C: tl.constexpr,
    B_SCALE_PTR, B_ZEROPOINT_PTR,
    BLOCK_ROW_PTR, BLOCK_COL_PTR,
    BLOCK_SIZE_PTR,
    GROUPS_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    SWITCH_TO_GEMM: tl.constexpr,
    GEMM_SWITCH_K: tl.constexpr,
    M_REMAINDER: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B is a quantized matrix in int4 format,
    C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_mask = (offs_am[:, None] < M)
    if not B_ROW_MAJOR:
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    else:
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b_mask = (offs_bn[None, :] < N)

    block_id = tl.load(BLOCK_ROW_PTR + pid_m)
    block_size = tl.load(BLOCK_SIZE_PTR + pid_m)
    start_n = tl.load(BLOCK_COL_PTR + block_id * M + offs_am[:, None]).to(tl.int32)
    end_n = start_n + block_size
    b_mask &= (offs_bn[None, :] >= start_n) & (offs_bn[None, :] < end_n)

    if not USE_INITIAL_C:
        c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    elif pid_m * num_pid_n + pid_n < M_REMAINDER * N:
        c = tl.load(C + pid_m * stride_cm + pid_n * stride_cn + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_cn)
    else:
        c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    k_remainder_mask = offs_k < k_remainder
    k_remainder_b_mask = k_remainder_mask[:, None] & b_mask

    for k in range(0, K, BLOCK_SIZE_K):
        b_int4_ptr = b_ptrs
        b_scale_ptr = B_SCALE_PTR + k // GROUP_SIZE_K
        b_zeropoint_ptr = B_ZEROPOINT_PTR + k // GROUP_SIZE_K
        if not B_ROW_MAJOR:
            b_int4_ptr = b_ptrs
        else:
            b_int4_ptr = b_ptrs
        b_int32 = tl.load(b_int4_ptr, mask=k_remainder_b_mask, other=0).to(tl.int32)
        b_dequantized = (b_int32 - tl.load(b_zeropoint_ptr)) * tl.load(b_scale_ptr)

        if SWITCH_TO_GEMM and k >= GEMM_SWITCH_K:
            a = tl.load(a_ptrs, mask=a_mask, other=0)
            c += tl.dot(a, b_dequantized, allow_tf32=True)
        else:
            a = tl.load(a_ptrs, mask=a_mask, other=0)
            c += tl.dot(a, b_dequantized, allow_tf32=True)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if STORE_FINAL_C:
        c = c.to(C.dtype.element_ty)
        tl.store(C + pid_m * stride_cm + pid_n * stride_cn + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_cn, c)
    else:
        c = c.to(C.dtype.element_ty)
        tl.store(C + pid_m * stride_cm + pid_n * stride_cn + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_cn, c, mask=a_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'GROUPS_M': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'GROUPS_M': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256,
