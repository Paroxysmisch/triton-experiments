import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=4),
    ],
    key=['K'],
)
@triton.jit
def quantize_int8_perrow_kernel(
    fpa_ptr,
    a_ptr,
    as_ptr,
    M, K,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    
    # Compute max value for the row
    max_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE):
        idxs = k + offs
        mask = idxs < K
        vals = tl.load(fpa_ptr + row_idx * K + idxs, mask=mask, other=0)
        abs_vals = tl.abs(vals)
        current_max = tl.max(abs_vals, axis=0)
        max_val = tl.maximum(max_val, current_max)
    row_max = tl.max(max_val, axis=0)
    
    # Quantize and store
    scale = 127.0 / row_max
    for k in range(0, K, BLOCK_SIZE):
        idxs = k + offs
        mask = idxs < K
        vals = tl.load(fpa_ptr + row_idx * K + idxs, mask=mask, other=0)
        quantized = vals * scale
        quantized_int8 = tl.libdevice.llrint(quantized).to(tl.int8)
        tl.store(a_ptr + row_idx * K + idxs, quantized_int8, mask=mask)
    tl.store(as_ptr + row_idx, row_max)

def quantize_int8_perrow(fpa: torch.Tensor):
    M, K = fpa.shape
    a = torch.empty_like(fpa, dtype=torch.int8)
    as_ = torch.empty(M, device=fpa.device, dtype=torch.float16)
    grid = (M,)
    quantize_int8_perrow_kernel[grid](fpa, a, as_, M, K, BLOCK_SIZE=min(triton.next_power_of_two(K), 4096))
    return a, as_

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 2}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for _ in range(K // (BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - _ * BLOCK_SIZE_K * SPLIT_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - _ * BLOCK_SIZE_K * SPLIT_K, other=0)
        a_i8 = a.to(tl.int8)
        b_i8 = b.to(tl.int8)
        accumulator += tl.dot(a_i8, b_i8, out_dtype=tl.int32)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    
    as_vals = tl.load(as_ptr + offs_m, mask=offs_m < M, other=0)
    bs_vals = tl.load(bs_ptr + offs_n, mask=offs_n < N, other=0)
    c = accumulator.to(tl.float32) * as_vals[:, None] * bs_vals[None, :]
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul_int8(a: torch.Tensor, b: torch.Tensor, as_: torch.Tensor, bs: torch.Tensor):
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Incompatible dimensions"
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128), 1)
    matmul_kernel[grid](
        a, b, c, as_, bs,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        SPLIT_K=1,
    )
    return c

def matmul_quantize_int8(fpa: torch.Tensor, fpb: torch.Tensor):
    a, as_ = quantize_int8_perrow(fpa)
    b, bs = quantize_int8_perrow(fpb.T)  # Transpose for column-wise quantization
    bs = bs.view(-1, 1)  # Reshape for broadcasting
    return matmul_int8(a, b.T, as_, bs.squeeze())  # Transpose B back

def quantize_int8(x: torch.Tensor, axis: int):
    if axis == 1:
        return quantize_int8_perrow(x)
    else:
        x_t = x.T
        quantized_t, scales_t = quantize_int8_perrow(x_t)
        return quantized_t.T, scales_t
