import torch
import triton
import triton.language as tl

@triton.jit
def softmax_mul_kernel(input, other, output, M, stride, K, CACHE_KEY_M, CACHE_KEY_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    if BLOCK_N == 1:
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        other_offs = offs_m
    else:
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        other_offs = offs_m[:, None] * stride + offs_n[None, :]
    input_ptrs = input + other_offs
    input_ptrs += off_hz * K
    other_ptrs = other + other_offs
    if dtype == torch.float16:
        input_vals = tl.load(input_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < K), other=0.0).to(tl.float32)
        other_vals = tl.load(other_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < K), other=0.0).to(tl.float32)
    else:
        input_vals = tl.load(input_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < K), other=0.0)
        other_vals = tl.load(other_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < K), other=0.0)
    max_vals = tl.max(input_vals, axis=1)
    exp_input = tl.exp(input_vals - max_vals[:, None])
    sum_input = tl.sum(exp_input, axis=1)
    softmax_output = exp_input / sum_input[:, None]
    output_ptrs = output + other_offs
    output_ptrs += off_hz * K
    tl.store(output_ptrs, softmax_output * other_vals, mask=(offs_m[:, None] < M) & (offs_n[None, :] < K))

def softmax_mul(input, other, dim, dtype=None, out=None) -> torch.Tensor:
    if dtype is None:
        dtype = input.dtype
    if out is None:
        out = torch.empty_like(input, dtype=dtype)
    assert input.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert other.dim() >= 2, "Other tensor must have at least 2 dimensions"
    assert input.shape == other.shape, "Input and other tensor must have the same shape"
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    N = 1
    M = 1
    for i in range(0, dim):
        N *= input.shape[i]
        M *= other.shape[i]
    for i in range(dim + 1, input.ndim):
        N *= input.shape[i]
    BLOCK_M = 128
    BLOCK_N = 128
    if N <= 128:
        BLOCK_M = N
        BLOCK_N = 1
    if N > 128:
        BLOCK_N = 128
    if N <= 64:
        BLOCK_M = N
        BLOCK_N = 1
    if N > 2048:
        BLOCK_M = 128
    if N <= 2048:
        BLOCK_M = 256
    if N > 4096:
        BLOCK_M = 512
    if N <= 4096:
        BLOCK_M = 1024
    if N > 8192:
        BLOCK_M = 2048
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    CACHE_KEY_M = M
    CACHE_KEY_K = N
    softmax_mul_kernel[grid](input, other, out, M, N, N, CACHE_KEY_M, CACHE_KEY_K, BLOCK_M, BLOCK_N)
    return out
