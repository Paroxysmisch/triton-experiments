import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_kernel(
    input1_ptr, input2_ptr, output_ptr,
    B, N, M, P,
    dropout_p, eps, training, approximate,
    stride_input1_b, stride_input1_n, stride_input1_m,
    stride_input2_b, stride_input2_m, stride_input2_p,
    stride_output_b, stride_output_n, stride_output_p,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_P: tl.constexpr,
):
    # Matrix multiplication
    pid = tl.program_id(axis=0)
    bid = pid // (N * P)
    nid = (pid % (N * P)) // P
    pid_p = pid % P

    offs_input1 = bid * stride_input1_b + nid * stride_input1_n + tl.arange(0, BLOCK_SIZE_M) * stride_input1_m
    offs_input2 = bid * stride_input2_b + tl.arange(0, BLOCK_SIZE_M) * stride_input2_m + pid_p * stride_input2_p
    offs_output = bid * stride_output_b + nid * stride_output_n + pid_p * stride_output_p

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
    for m in range(0, M, BLOCK_SIZE_M):
        a = tl.load(input1_ptr + offs_input1 + m, mask=m + tl.arange(0, BLOCK_SIZE_M) < M, other=0.0)
        b = tl.load(input2_ptr + offs_input2 + m, mask=m + tl.arange(0, BLOCK_SIZE_M) < M, other=0.0)
        acc += tl.dot(a, b)

    # RMS normalization
    mean = tl.sum(acc * acc, axis=1) / P
    inv_std = 1.0 / tl.sqrt(mean + eps)
    acc = acc * inv_std[:, None]

    # GELU activation
    if approximate == 'none':
        acc = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
    else:
        acc = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))

    # Dropout
    if training:
        mask = tl.rand((BLOCK_SIZE_N, BLOCK_SIZE_P)) > dropout_p
        acc = tl.where(mask, acc / (1.0 - dropout_p), 0.0)

    tl.store(output_ptr + offs_output, acc)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_P': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_P': 64}, num_warps=2),
    ],
    key=['B', 'N', 'M', 'P'],
)
@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_kernel(
    input1_ptr, input2_ptr, output_ptr,
    B, N, M, P,
    dropout_p, eps, training, approximate,
    stride_input1_b, stride_input1_n, stride_input1_m,
    stride_input2_b, stride_input2_m, stride_input2_p,
    stride_output_b, stride_output_n, stride_output_p,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_P: tl.constexpr,
):
    # Matrix multiplication
    pid = tl.program_id(axis=0)
    bid = pid // (N * P)
    nid = (pid % (N * P)) // P
    pid_p = pid % P

    offs_input1 = bid * stride_input1_b + nid * stride_input1_n + tl.arange(0, BLOCK_SIZE_M) * stride_input1_m
    offs_input2 = bid * stride_input2_b + tl.arange(0, BLOCK_SIZE_M) * stride_input2_m + pid_p * stride_input2_p
    offs_output = bid * stride_output_b + nid * stride_output_n + pid_p * stride_output_p

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
    for m in range(0, M, BLOCK_SIZE_M):
        a = tl.load(input1_ptr + offs_input1 + m, mask=m + tl.arange(0, BLOCK_SIZE_M) < M, other=0.0)
        b = tl.load(input2_ptr + offs_input2 + m, mask=m + tl.arange(0, BLOCK_SIZE_M) < M, other=0.0)
        acc += tl.dot(a, b)

    # RMS normalization
    mean = tl.sum(acc * acc, axis=1) / P
    inv_std = 1.0 / tl.sqrt(mean + eps)
    acc = acc * inv_std[:, None]

    # GELU activation
    if approximate == 'none':
        acc = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
    else:
        acc = 0.5 * acc * (1.0 + tl.math.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))

    # Dropout
    if training:
        mask = tl.rand((BLOCK_SIZE_N, BLOCK_SIZE_P)) > dropout_p
        acc = tl.where(mask, acc / (1.0 - dropout_p), 0.0)

    tl.store(output_ptr + offs_output, acc)

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    B, N, M = input1.shape
    _, _, P = input2.shape

    if out is None:
        out = torch.empty((B, N, P), dtype=input1.dtype, device=input1.device)

    grid = (B * N * P // (128 * 128),)
    fused_bmm_rmsnorm_gelu_dropout_kernel[grid](
        input1, input2, out,
        B, N, M, P,
        dropout_p, eps, training, approximate,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
    )
    return out
