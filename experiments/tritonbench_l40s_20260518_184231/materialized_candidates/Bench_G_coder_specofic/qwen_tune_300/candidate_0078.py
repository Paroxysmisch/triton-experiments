import torch
import triton
import triton.language as tl
from .math import next_power_of_two
from .utils import calculate_settings

@triton.jit
def _int8_matmul_rowwise_dequantize(
    a_ptr,
    b_ptr,
    state_x_ptr,
    state_w_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    c_stride_m,
    c_stride_n,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    pid_sk = tl.program_id(axis=1)
    a_ptr = a_ptr + pid * BLOCK_M * K
    b_ptr = b_ptr + pid * BLOCK_M * K
    if SPLIT_K == 1:
        c_ptr = c_ptr + pid * BLOCK_M * N
    else:
        c_ptr = c_ptr + pid * BLOCK_M * N + pid_sk * BLOCK_N
    state_x_ptr = state_x_ptr + pid * BLOCK_M
    state_w_ptr = state_w_ptr + pid * BLOCK_M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if SPLIT_K == 1:
            a = tl.load(a_ptr + k * BLOCK_K * SPLIT_K)
            b = tl.load(b_ptr + k * BLOCK_K * SPLIT_K)
        else:
            a = tl.load(a_ptr + k * BLOCK_K * SPLIT_K + pid_sk * BLOCK_N)
            b = tl.load(b_ptr + k * BLOCK_K * SPLIT_K + pid_sk * BLOCK_N)
        acc += tl.dot(a, b, allow_tf32=False)

    scale = tl.load(state_x_ptr).to(tl.float32)
    bias = tl.load(bias_ptr).to(tl.float32)
    acc = (acc.to(tl.float32) * scale).to(tl.int32) + bias

    w = tl.load(state_w_ptr)
    acc = tl.where(acc <= 0, 0, acc)
    acc = (acc * w).to(tl.int32)

    if SPLIT_K == 1:
        tl.store(c_ptr, acc)
    else:
        tl.atomic_add(c_ptr, acc)

def int8_matmul_rowwise_dequantize(a, b, state_x, state_w, bias):
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    if bias.stride(0) > 1:
        bias = bias.contiguous()

    M, K = a.shape
    _, N = b.shape
    # alloc output
    c = torch.empty((M, N), device=device, dtype=torch.int32)
    # accumulator
    acc = torch.empty((M, N), device=device, dtype=torch.int32)
    # split k dimension for better utilization
    SPLIT_K = 8
    if K <= 2048:
        BLOCK_M, BLOCK_N, BLOCK_K = calculate_settings(K, tuning_k=K)
        BLOCK_M = next_power_of_two(BLOCK_M)
        BLOCK_N = next_power_of_two(BLOCK_N)
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = calculate_settings(K, tuning_k=2048)
        BLOCK_M = next_power_of_two(BLOCK_M)
        BLOCK_N = next_power_of_two(BLOCK_N)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        SPLIT_K if META["BLOCK_N"] == N else 1,
    )
    _int8_matmul_rowwise_dequantize[grid](
        a,
        b,
        state_x,
        state_w,
        bias,
        c,
        M,
        N,
        K,
        c.stride(0),
        c.stride(1),
        SPLIT_K=SPLIT_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c
