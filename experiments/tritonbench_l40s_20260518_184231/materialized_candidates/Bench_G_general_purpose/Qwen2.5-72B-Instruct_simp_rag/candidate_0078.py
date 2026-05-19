import triton
import triton.language as tl
import torch
from typing import Optional

# Triton kernel for int8 matrix multiplication with row-wise dequantization
@triton.jit
def _int8_matmul_rowwise_dequantize(
    A, B, state_x, state_w, C, bias, 
    M, N, K, 
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn, 
    stride_state_x, stride_state_w, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr, 
    SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = SPLIT_K * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m - first_pid_m, SPLIT_K)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // SPLIT_K

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs_k = tl.arange(0, BLOCK_K)

    A = A + (offs_am * stride_am + offs_k * stride_ak)
    B = B + (offs_k[:, None] * stride_bk + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # Dequantize using state_x and state_w
    state_x_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    state_w_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    state_x = tl.load(state_x + state_x_offs * stride_state_x)
    state_w = tl.load(state_w + state_w_offs * stride_state_w)

    acc = acc * state_x[:, None] * state_w[None, :]

    if bias is not None:
        bias_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias = tl.load(bias + bias_offs)
        acc += bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(C, acc.to(tl.float16))

# Wrapper function to launch the kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def int8_matmul_rowwise_dequantize(
    A: torch.Tensor, 
    B: torch.Tensor, 
    state_x: torch.Tensor, 
    state_w: torch.Tensor, 
    C: torch.Tensor, 
    bias: Optional[torch.Tensor] = None, 
    M: int = -1, 
    N: int = -1, 
    K: int = -1, 
    BLOCK_M: int = 128, 
    BLOCK_N: int = 128, 
    BLOCK_K: int = 32, 
    SPLIT_K: int = 1
):
    if M == -1:
        M = A.shape[0]
    if N == -1:
        N = B.shape[1]
    if K == -1:
        K = A.shape[1]

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * SPLIT_K, )
    _int8_matmul_rowwise_dequantize[grid](
        A, B, state_x, state_w, C, bias,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        state_x.stride(0), state_w.stride(0),
        BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K
    )
