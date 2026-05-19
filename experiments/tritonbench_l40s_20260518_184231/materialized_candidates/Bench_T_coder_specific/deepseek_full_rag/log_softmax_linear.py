import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_input_m, stride_input_k,
    stride_weight_n, stride_weight_k,
    stride_output_m, stride_output_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    A_BITS: tl.constexpr,
    B_BITS: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    has_bias: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = input_ptr + (offs_am[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
    b_ptrs = weight_ptr + (offs_bn[None, :] * stride_weight_n + offs_k[:, None] * stride_weight_k)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN_K:
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k, other=0.0).to(tl.float32)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k, other=0.0).to(tl.float32)
            accumulator += tl.dot(a, b)
        a = tl.load(a_ptrs, mask=True, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=True, other=0.0).to(tl.float32)
        accumulator += tl.dot(a, b)
    else:
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs + k * stride_input_k, mask=offs_k[None, :] < (K - k), other=0.0).to(tl.float32)
            b = tl.load(b_ptrs + k * stride_weight_k, mask=offs_k[:, None] < (K - k), other=0.0).to(tl.float32)
            accumulator += tl.dot(a, b)
        a = tl.load(a_ptrs + (K - 1) * stride_input_k, mask=True, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs + (K - 1) * stride_weight_k, mask=True, other=0.0).to(tl.float32)
        accumulator += tl.dot(a, b)

    if has_bias:
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs, mask=offs_bn < N, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    if OUTPUT_DTYPE in (tl.float16, tl.bfloat16):
        accumulator = accumulator.to(OUTPUT_DTYPE)
    else:
        accumulator = accumulator.to(tl.float16)

    output_ptrs = output_ptr + stride_output_m * offs_am[:, None] + stride_output_n * offs_bn[None, :]
    tl.store(output_ptrs, accumulator)


def linear(input, weight, bias=None, dim=-1, dtype=None):
    shape = input.shape
    if dim != -1:
        input = input.transpose(dim, -1).contiguous()
        input_shape = input.shape
        input = input.view(-1, input_shape[-1])
        N, K = input.shape
        M = N * input_shape[dim]
    else:
        N, K = input.shape
        input_shape = None
        M = N
    N, K = input.shape
    assert K == weight.shape[1]
    if bias is not None:
        assert bias.shape[0] == weight.shape[0]
    if dtype is None:
        dtype = input.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("linear only supports fp16 and bf16")
    if input.dtype != dtype:
        input = input.to(dtype)
    if weight.dtype != dtype:
        weight = weight.to(dtype)
    if bias is not None and bias.dtype != dtype:
        bias = bias.to(dtype)
    N, K = input.shape
    assert K == weight.shape[1]
    if bias is not None:
        assert bias.shape[0] == weight.shape[0]
    if dim != -1:
        N, K = input.shape
        M = N * input_shape[dim]
    else:
        M = N
    N, K = input.shape
    assert K == weight.shape[1]
    if bias is not None:
        assert bias.shape[0] == weight.shape[0]
    if dim != -1:
        N, K = input.shape
        M = N * input_shape[dim]
    else:
        M = N
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    output = torch.empty((M, N), device=input.device, dtype=dtype)
    bias_dtype = None if bias is None else bias.dtype
    kernel = linear_kernel[grid]
    kernel[(M, N)](
        input, weight, bias, output,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        has_bias=bias is not None,
        A_BITS=input.element_size() * 8,
        B_BITS=weight.element_size() * 8,
        GROUP_M=8,
        EVEN_K=K % (32 * 1024) == 0,
        BLOCK_K=32 * 1024,
        OUTPUT_DTYPE=dtype
    )
    if dim != -1:
        output = output.transpose(dim, -1).contiguous()
    return output
