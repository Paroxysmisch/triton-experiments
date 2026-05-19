import triton
import triton.language as tl
import torch
from typing import Optional

@triton.jit
def _int8_matmul_rowwise_dequantize(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    state_x_ptr, state_w_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, ACC_TYPE: tl.constexpr,
    IS_BIAS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = num_pid_m - first_pid_m
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, K, BLOCK_K * SPLIT_K):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_K * SPLIT_K * stride_bk

    # Dequantize
    state_x = tl.load(state_x_ptr + offs_am, mask=offs_am < M, other=0.0)
    state_w = tl.load(state_w_ptr + offs_am, mask=offs_am < M, other=0.0)
    acc = acc * state_x[:, None] * state_w[:, None]

    # Add bias if present
    if IS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        acc += bias[None, :]

    # Store the result
    c_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)

def int8_matmul_rowwise_dequantize(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    state_x: torch.Tensor = None,
    state_w: torch.Tensor = None,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 32,
    SPLIT_K: int = 1,
    ACC_TYPE: str = 'tl.int32'
):
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    assert C.dtype == torch.float32
    M, K = A.shape
    N = B.shape[1]
    assert B.shape[0] == K
    if bias is not None:
        assert bias.shape == (N,)
    if state_x is not None:
        assert state_x.shape == (M,)
    if state_w is not None:
        assert state_w.shape == (M,)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    _int8_matmul_rowwise_dequantize[grid](
        A, B, C, bias,
        state_x, state_w,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        bias.stride(0) if bias is not None else 0,
        BLOCK_M, BLOCK_N, BLOCK_K,
        SPLIT_K, ACC_TYPE,
        IS_BIAS=bias is not None
    )
