import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _int8_matmul_rowwise_dequantize(
    A, B, C, state_x, state_w, bias, M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_xm, stride_wn,
    stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_wg = num_pid_m * num_pid_n
    num_wg = tl.cdiv(K, BLOCK_K * SPLIT_K)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_g = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_g * BLOCK_K + tl.arange(0, BLOCK_K)

    A = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    state_x = state_x + (offs_m * stride_xm)
    state_w = state_w + (offs_n * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K * SPLIT_K):
        a = tl.load(A)
        b = tl.load(B)
        a = a * tl.load(state_x)[:, None]
        b = b * tl.load(state_w)[None, :]
        acc += tl.dot(a, b)

    acc = acc.to(tl.float16)

    if bias is not None:
        bias = bias + (offs_n * stride_bn)
        bias = tl.load(bias)
        acc += bias[None, :]

    C = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C, acc)

import torch
import triton
import triton.language as tl

@triton.jit
def _int8_matmul_rowwise_dequantize(
    A, B, C, state_x, state_w, bias, M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_xm, stride_wn,
    stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    # The kernel function is defined above
    pass

def int8_matmul_rowwise_dequantize(A, B, state_x, state_w, bias=None):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), triton.cdiv(K, META['BLOCK_K'] * META['SPLIT_K']))

    _int8_matmul_rowwise_dequantize[grid](
        A, B, C, state_x, state_w, bias, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        state_x.stride(0), state_w.stride(0),
        B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, SPLIT_K=1,
    )

    return C

import torch

# Example inputs
M, K, N = 1024, 1024, 1024
A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device='cuda')
state_x = torch.randn(M, dtype=torch.float32, device='cuda')
state_w = torch.randn(N, dtype=torch.float32, device='cuda')
bias = torch.randn(N, dtype=torch.float16, device='cuda')

# Perform the matrix multiplication with row-wise dequantization
C = int8_matmul_rowwise_dequantize(A, B, state_x, state_w, bias)

print(C)
