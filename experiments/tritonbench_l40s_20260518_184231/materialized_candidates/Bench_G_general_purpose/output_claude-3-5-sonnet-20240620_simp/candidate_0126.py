import triton
import triton.language as tl
import torch
import math

# Utility functions for quantization
@torch.jit.script
def quantize_int4(x: torch.Tensor, group_size: int = 128):
    # Reshape for group-wise quantization
    x_2d = x.reshape(-1, group_size)
    num_groups = x_2d.shape[0]
    
    # Calculate scales and zero points
    max_vals = x_2d.amax(dim=1)
    min_vals = x_2d.amin(dim=1)
    scales = (max_vals - min_vals) / 15
    zeros = min_vals
    
    # Quantize to int4
    x_scaled = ((x_2d - zeros.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15)
    
    # Pack 8 int4 values into int32
    x_packed = torch.zeros((num_groups, group_size // 8), dtype=torch.int32, device=x.device)
    for i in range(8):
        x_packed |= (x_scaled[:, i::8].to(torch.int32) & 0xF) << (i * 4)
    
    return x_packed, scales, zeros

# Triton kernel for dequantization
@triton.jit
def dequantize_kernel(
    b_ptr, fpb_ptr, scales_ptr, zeros_ptr,
    K, N,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate block start indices
    k_start = (pid // (N // BLOCK_SIZE_N)) * BLOCK_SIZE_K
    n_start = (pid % (N // BLOCK_SIZE_N)) * BLOCK_SIZE_N
    
    # Load scales and zeros for this block
    scale = tl.load(scales_ptr + k_start // BLOCK_SIZE_K)
    zero = tl.load(zeros_ptr + k_start // BLOCK_SIZE_K)
    
    # Process block
    for k in range(BLOCK_SIZE_K // 8):
        for n in range(BLOCK_SIZE_N):
            # Load packed int4 value
            packed = tl.load(b_ptr + (k_start // 8 + k) * N + n_start + n)
            
            # Unpack and dequantize 8 values
            for i in range(8):
                val_int4 = (packed >> (i * 4)) & 0xF
                val_fp = val_int4 * scale + zero
                tl.store(fpb_ptr + (k_start + k * 8 + i) * N + n_start + n, val_fp)

# Triton kernel for matrix multiplication with int4 input
@triton.jit
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}),
    ],
    key=['M', 'N', 'K'],
)
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        # Load scales and zeros for this block
        scale = tl.load(scales_ptr + k // BLOCK_SIZE_K)
        zero = tl.load(zeros_ptr + k // BLOCK_SIZE_K)
        
        # Load and process A block
        a = tl.load(a_ptrs)
        
        # Load and process B block (with int4 unpacking)
        b_packed = tl.load(b_ptrs)
        b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        
        for i in range(8):
            mask = 0xF << (i * 4)
            vals_int4 = (b_packed & mask) >> (i * 4)
            b[i::8, :] = vals_int4 * scale + zero
        
        # Compute matrix multiplication for this block
        accumulator += tl.dot(a, b)
        
        # Update pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk
    
    # Store result
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c)

# Wrapper functions
def dequantize_int4(b_packed, scales, zeros):
    K = scales.shape[0] * 8
    N = b_packed.shape[1]
    
    # Output tensor
    fpb = torch.empty((K, N), dtype=torch.float16, device=b_packed.device)
    
    def grid(meta):
        return (triton.cdiv(K, 64) * triton.cdiv(N, 128),)
    
    dequantize_kernel[grid](
        b_packed, fpb, scales, zeros,
        K, N,
        BLOCK_SIZE_K=64, BLOCK_SIZE_N=128,
    )
    
    return fpb

def matmul_dequantize_int4_s1(a, b_packed, scales, zeros):
    M, K = a.shape
    K_packed, N = b_packed.shape
    assert K == K_packed * 8, "Incompatible dimensions"
    
    # Output tensor
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    matmul4_kernel[grid](
        a, b_packed, c, scales, zeros,
        M, N, K,
        a.stride(0), a.stride(1),
        b_packed.stride(0), b_packed.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c
