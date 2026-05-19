import triton
import triton.language as tl

@triton.jit
def _int8_matmul_rowwise_dequantize(
    A, B, C, bias, state_x_ptr, state_w_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ADD_BIAS: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = SPLIT_K * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m - first_pid_m, BLOCK_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // SPLIT_K

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    A_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_am, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    B_block_ptr = tl.make_block_ptr(
        base=B,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, offs_bn),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, num_pid_k):
        a = tl.load(A_block_ptr)
        b = tl.load(B_block_ptr)
        acc += tl.dot(a, b)
        A_block_ptr = tl.advance(A_block_ptr, [0, BLOCK_K])
        B_block_ptr = tl.advance(B_block_ptr, [BLOCK_K, 0])

    state_x = tl.load(state_x_ptr + pid_m)
    state_w = tl.load(state_w_ptr + pid_n)
    acc = acc * state_x * state_w

    if ADD_BIAS:
        bias_block_ptr = tl.make_block_ptr(
            base=bias,
            shape=(N,),
            strides=(stride_bias_n,),
            offsets=(offs_bn,),
            block_shape=(BLOCK_N,),
            order=(0,)
        )
        bias_val = tl.load(bias_block_ptr)
        acc += bias_val

    C_block_ptr = tl.make_block_ptr(
        base=C,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offs_am, offs_bn),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    if SPLIT_K == 1:
        tl.store(C_block_ptr, acc.to(tl.float32))
    else:
        tl.atomic_add(C_block_ptr, acc.to(tl.float32))

import triton
import triton.language as tl
import torch

def int8_matmul_rowwise_dequantize(A, B, bias, state_x, state_w, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, ADD_BIAS):
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    state_x = state_x.contiguous()
    state_w = state_w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Allocate output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Compute grid size
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']) * meta['SPLIT_K'],
    )

    # Launch kernel
    _int8_matmul_rowwise_dequantize[grid](
        A, B, C, bias, state_x, state_w,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        bias.stride(0) if bias is not None else 0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        ADD_BIAS=ADD_BIAS
    )

    return C
