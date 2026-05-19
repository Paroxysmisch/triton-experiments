import math
import torch
import triton
import triton.language as tl

# Triton kernel for rowwise quantization
@triton.jit
def quantize_int8_perrow_kernel(
    fpa_ptr, a_ptr, as_ptr, M, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr, num_warps: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE_M
    offsets = tl.arange(0, BLOCK_SIZE_K)
    row_mask = row_start + offsets < M

    # Load the input elements
    fpa = tl.load(fpa_ptr + row_start * K + offsets, mask=row_mask, other=0.0)

    # Calculate the absolute maximum value for normalization
    abs_fpa = tl.abs(fpa)
    max_val = tl.max(tl.where(row_mask, abs_fpa, 0), axis=0)
    
    # Quantize the input elements to int8
    quantized = tl.libdevice.llrint(127.0 * (fpa / max_val))
    
    # Store the quantized output and max values
    tl.store(a_ptr + row_start * K + offsets, quantized, mask=row_mask)
    tl.store(as_ptr + pid, max_val)

def quantize_int8_perrow(fpa: torch.Tensor):
    M, K = fpa.shape
    a = torch.empty_like(fpa, dtype=torch.int8)
    as_ = torch.empty(M, device=fpa.device, dtype=torch.float32)

    # Define grid configuration
    grid = lambda meta: (M,)
    
    # Launch the Triton kernel
    quantize_int8_perrow_kernel[grid](
        fpa, a, as_, M, K,
        BLOCK_SIZE_M=128, BLOCK_SIZE_K=128,
        num_stages=2, num_warps=4
    )
    return a, as_

# Triton kernel for matrix multiplication with quantized int8 matrices
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr, num_warps: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptr + block_start_m * K + k, mask=block_start_m + tl.arange(0, BLOCK_SIZE_M) < M, other=0.0)
        b = tl.load(b_ptr + block_start_n + k * N, mask=block_start_n + tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
        
        acc += tl.dot(a, b)

    scale_a = tl.load(as_ptr + block_start_m, mask=block_start_m + tl.arange(0, BLOCK_SIZE_M) < M)
    scale_b = tl.load(bs_ptr + block_start_n, mask=block_start_n + tl.arange(0, BLOCK_SIZE_N) < N)

    acc = acc * (scale_a[:, None] * scale_b[None, :])
    
    tl.store(c_ptr + block_start_m * N + block_start_n, acc, mask=(block_start_m + tl.arange(0, BLOCK_SIZE_M) < M)[:, None] & (block_start_n + tl.arange(0, BLOCK_SIZE_N) < N)[None, :])

def matmul_int8(a: torch.Tensor, b: torch.Tensor, as_: torch.Tensor, bs_: torch.Tensor):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda meta: (M // meta['BLOCK_SIZE_M'], N // meta['BLOCK_SIZE_N'], meta['SPLIT_K'])

    matmul_kernel[grid](
        a, b, c, as_, bs_,
        M, N, K,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        num_stages=2, num_warps=4, SPLIT_K=1
    )
    return c

def matmul_quantize_int8(fpa: torch.Tensor, fpb: torch.Tensor):
    a, as_ = quantize_int8_perrow(fpa)
    b, bs_ = quantize_int8_perrow(fpb)
    c = matmul_int8(a, b, as_, bs_)
    return c
