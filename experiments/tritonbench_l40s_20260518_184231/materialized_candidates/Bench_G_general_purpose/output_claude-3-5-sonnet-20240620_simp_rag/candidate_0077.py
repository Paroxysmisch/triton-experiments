import math
import torch
import triton
import triton.language as tl

@triton.jit
def quantize_int8_perrow_kernel(
    x_ptr, output_ptr, scale_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * N
    
    # Load row
    row = tl.load(x_ptr + row_start + tl.arange(0, BLOCK_SIZE))
    
    # Compute max absolute value
    row_abs = tl.abs(row)
    max_val = tl.max(row_abs, axis=0)
    
    # Compute scale
    scale = max_val / 127.0
    
    # Quantize
    quantized = tl.libdevice.llrint(row / scale)
    
    # Store results
    tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), quantized)
    tl.store(scale_ptr + pid, scale)

def quantize_int8_perrow(x: torch.Tensor):
    M, N = x.shape
    output = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    
    grid = (M,)
    quantize_int8_perrow_kernel[grid](x, output, scale, M, N, BLOCK_SIZE=N)
    
    return output, scale

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        scale_a = tl.load(scale_a_ptr + offs_am)
        scale_b = tl.load(scale_b_ptr + offs_bn)
        
        a = a.to(tl.float32) * scale_a[:, None]
        b = b.to(tl.float32) * scale_b[None, :]
        
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    
    c_ptrs = c_ptr + offs_am[:, None] * N + offs_bn[None, :]
    tl.store(c_ptrs, accumulator)

def matmul_quantize_int8(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    _, N = b.shape
    
    a_quantized, scale_a = quantize_int8_perrow(a)
    b_quantized, scale_b = quantize_int8_perrow(b)
    
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    
    grid = (triton.cdiv(M, 32) * triton.cdiv(N, 32),)
    matmul_kernel[grid](
        a_quantized, b_quantized, c,
        scale_a, scale_b,
        M, N, K,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
    )
    
    return c

def matmul_int8(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor):
    M, K = a.shape
    _, N = b.shape
    
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    
    grid = (triton.cdiv(M, 32) * triton.cdiv(N, 32),)
    matmul_kernel[grid](
        a, b, c,
        scale_a, scale_b,
        M, N, K,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
    )
    
    return c

@triton.jit
def quantize_int8_kernel(
    x_ptr, output_ptr, scale_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
    axis: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if axis == 0:
        col_start = pid * BLOCK_SIZE
        col = tl.load(x_ptr + col_start + tl.arange(0, BLOCK_SIZE))
        col_abs = tl.abs(col)
        max_val = tl.max(col_abs, axis=0)
        scale = max_val / 127.0
        quantized = tl.libdevice.llrint(col / scale)
        tl.store(output_ptr + col_start + tl.arange(0, BLOCK_SIZE), quantized)
        tl.store(scale_ptr + pid, scale)
    else:
        row_start = pid * N
        row = tl.load(x_ptr + row_start + tl.arange(0, BLOCK_SIZE))
        row_abs = tl.abs(row)
        max_val = tl.max(row_abs, axis=0)
        scale = max_val / 127.0
        quantized = tl.libdevice.llrint(row / scale)
        tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), quantized)
        tl.store(scale_ptr + pid, scale)

def quantize_int8(x: torch.Tensor, axis: int = 0):
    M, N = x.shape
    output = torch.empty_like(x, dtype=torch.int8)
    
    if axis == 0:
        scale = torch.empty(N, dtype=torch.float32, device=x.device)
        grid = (N,)
        quantize_int8_kernel[grid](x, output, scale, M, N, BLOCK_SIZE=M, axis=0)
    else:
        scale = torch.empty(M, dtype=torch.float32, device=x.device)
        grid = (M,)
        quantize_int8_kernel[grid](x, output, scale, M, N, BLOCK_SIZE=N, axis=1)
    
    return output, scale
