import torch
import triton
import triton.language as tl

# Quantization Kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
    ],
    key=['M', 'K']
)
@triton.jit
def quantize_int8_perrow_kernel(
    fpa_ptr, a_ptr, as_ptr,
    M, K,
    stride_fpam, stride_fpak,
    stride_am, stride_ak,
    stride_asm,
    BLOCK_SIZE_K: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = fpa_ptr + row_idx * stride_fpam
    
    # Find max abs value in row
    max_abs = tl.zeros((1,), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        cols = k + tl.arange(0, BLOCK_SIZE_K)
        mask = cols < K
        vals = tl.load(row_start + cols * stride_fpak, mask=mask, other=0.0)
        curr_abs = tl.abs(vals)
        curr_max = tl.max(curr_abs, axis=0)
        max_abs = tl.maximum(max_abs, curr_max)
    
    scale = max_abs / 127.0
    tl.store(as_ptr + row_idx * stride_asm, scale)
    
    # Quantize and store
    for k in range(0, K, BLOCK_SIZE_K):
        cols = k + tl.arange(0, BLOCK_SIZE_K)
        mask = cols < K
        vals = tl.load(row_start + cols * stride_fpak, mask=mask, other=0.0)
        quantized = (vals / scale).to(tl.int8)
        tl.store(a_ptr + row_idx * stride_am + cols * stride_ak, quantized, mask=mask)

def quantize_int8_perrow(fpa: torch.Tensor):
    M, K = fpa.shape
    a = torch.empty_like(fpa, dtype=torch.int8)
    as_ = torch.empty(M, device=fpa.device, dtype=torch.float32)
    
    grid = lambda _: (M,)
    quantize_int8_perrow_kernel[grid](
        fpa, a, as_,
        M, K,
        fpa.stride(0), fpa.stride(1),
        a.stride(0), a.stride(1),
        as_.stride(0),
        BLOCK_SIZE_K=128  # Default value, autotune may override
    )
    return a, as_

# Matrix Multiplication Kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, as_ptr, bs_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
    for _ in range(0, K // BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Load scaling factors
    scale_a = tl.load(as_ptr + offs_m * stride_asm, mask=offs_m < M)
    scale_b = tl.load(bs_ptr + offs_n * stride_bsn, mask=offs_n < N)
    
    # Apply scaling and store
    c = (acc.to(tl.float32) * scale_a[:, None]) * scale_b[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul_int8(a: torch.Tensor, as_: torch.Tensor, b: torch.Tensor, bs_: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_SIZE_M']),
            triton.cdiv(N, meta['BLOCK_SIZE_N']),
        )
    
    matmul_kernel[grid](
        a, b, as_, bs_, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        as_.stride(0), bs_.stride(0),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32
    )
    return c

# Combined Quantization and Matmul
def matmul_quantize_int8(fpa: torch.Tensor, b: torch.Tensor, bs: torch.Tensor):
    a, as_ = quantize_int8_perrow(fpa)
    return matmul_int8(a, as_, b, bs)

# General Quantization Function
def quantize_int8(matrix: torch.Tensor, axis: int = -1):
    if axis == 1 or axis == -1:
        return quantize_int8_perrow(matrix)
    else:
        raise NotImplementedError("Only row-wise quantization implemented")
