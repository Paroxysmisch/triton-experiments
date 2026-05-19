import math
import torch
import triton
import triton.language as tl
from typing import Optional

# Triton kernels
@triton.jit
def quantize_int8_perrow_kernel(
    fpa_ptr,
    a_ptr,
    as_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    **meta
):

    pid = tl.program_id(axis=0)
    warp_id = pid // num_stages
    lane_id = tl.lane_id()
    fpa_base = fpa_ptr + warp_id * K * 2
    a_base = a_ptr + warp_id * K
    as_base = as_ptr + warp_id

    accumulator = tl.zeros((num_stages,), dtype=tl.float16)
    max_value = tl.zeros((num_stages,), dtype=tl.float16)

    for stage in range(num_stages):
        if stage == num_stages - 1 and num_warps % 2 == 1:
            break
        offset = (2 * stage * num_warps * K, 2 * (stage + 1) * num_warps * K)
        fpa_sub = tl.load(fpa_base + offset, mask=lane_id < K)
        abs_value = tl.abs(fpa_sub)
        max_value_sub = tl.max(abs_value, axis=0)
        max_value = tl.max(max_value_sub, max_value)

    scale = tl.max(max_value)
    as_val = scale

    for stage in range(num_stages):
        offset = (2 * stage * num_warps * K, 2 * (stage + 1) * num_warps * K)
        fpa_sub = tl.load(fpa_base + offset, mask=lane_id < K)
        quantized_sub = tl.floor(fpa_sub / scale * 127)
        tl.store(a_base + offset, quantized_sub, mask=lane_id < K)

    tl.store(as_base, as_val)

@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    as_ptr,
    bs_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    **meta
):

    pid = tl.program_id(axis=0)
    warp_id = pid // BLOCK_SIZE_N
    lane_id = tl.lane_id()
    A_base = A_ptr + warp_id * K
    B_base = B_ptr
    C_base = C_ptr + warp_id * N

    if pid == 0:
        accumulator = tl.zeros((M,), dtype=tl.float32)
    accumulator = tl.allocate_shared(accumulator, scope=tl.Scope.WORKER)

    for k_chunk in range(K // SPLIT_K):
        k_start = k_chunk * SPLIT_K
        k_end = (k_chunk + 1) * SPLIT_K if k_chunk < K // SPLIT_K - 1 else K
        A_sub = tl.load(A_base + k_start, mask=lane_id < k_end - k_start)
        B_sub = tl.load(B_base + k_start, mask=lane_id < k_end - k_start)
        as_sub = tl.load(as_ptr + k_start)
        bs_sub = tl.load(bs_ptr + k_start)

        accumulator += A_sub * B_sub * (1.0 / (as_sub * bs_sub))

    for n in range(N):
        C = accumulator[n]
        tl.store(C_base + n, C, mask=lane_id < N)


# Python wrappers
def quantize_int8_perrow(fpa: torch.Tensor, num_stages: Optional[int] = 4, num_warps: Optional[int] = 16):
    num_warps = num_warps if num_warps > 0 else fpa.shape[0] // 2
    num_stages = num_stages if num_stages > 0 else 1
    a = torch.empty_like(fpa, dtype=torch.int8)
    as_out = torch.empty((fpa.shape[0],), dtype=torch.float16)
    M, K = fpa.shape
    grid = lambda meta: (meta["group_size"],)
    quantize_int8_perrow_kernel[grid](
        fpa.data_ptr(),
        a.data_ptr(),
        as_out.data_ptr(),
        M=M,
        K=K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return a, as_out

def matmul_quantize_int8(A: torch.Tensor, B: torch.Tensor, as_A: torch.Tensor, bs_B: torch.Tensor):
    A, as_A = quantize_int8_perrow(A)
    B, bs_B = quantize_int8_perrow(B)
    C = torch.zeros_like(A[0], dtype=torch.float32)
    matmul_kernel(
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        as_A.data_ptr(),
        bs_B.data_ptr(),
        *A.shape,
        *B.shape,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        SPLIT_K=16,
    )
    return C

def quantize_int8(matrix: torch.Tensor, axis: Optional[int] = 1):
    if axis == 1:
        return quantize_int8_perrow(matrix)
    else:
        raise ValueError("Unsupported axis for matrix quantization.")
