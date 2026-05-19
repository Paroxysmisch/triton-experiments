import torch
import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n, stride_zeros_z,
    GROUP_SIZE: tl.constexpr,
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
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0)
        
        # Unpack 4-bit values
        shifts = tl.arange(0, 8) * 4
        b_packed_exp = tl.expand(b_packed, (BLOCK_SIZE_K//8, BLOCK_SIZE_N, 8))
        b_vals = (b_packed_exp >> shifts[None, None, :]) & 0xF
        b_vals = tl.reshape(b_vals, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        
        # Compute dequantization parameters
        global_k = k + offs_k[:, None]
        group_idx = global_k // GROUP_SIZE
        k_in_group = global_k % GROUP_SIZE
        
        # Load scales
        scale_ptrs = scales_ptr + group_idx * stride_scales_g + offs_n[None, :] * stride_scales_n
        scales = tl.load(scale_ptrs, mask=b_mask, other=0.0)
        
        # Load and unpack zeros
        zero_idx = k_in_group // 8
        zero_shift = (k_in_group % 8) * 4
        zero_ptrs = zeros_ptr + group_idx * stride_zeros_g + offs_n[None, :] * stride_zeros_n + zero_idx * stride_zeros_z
        zeros_packed = tl.load(zero_ptrs, mask=b_mask, other=0)
        zeros = (zeros_packed >> zero_shift) & 0xF
        
        # Dequantize and accumulate
        b_dequant = (b_vals - zeros) * scales
        acc += tl.dot(a.to(tl.float32), b_dequant.to(tl.float32))
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K//8) * stride_bk
    
    # Store output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_am + offs_cn[None, :] * stride_ak
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

def matmul_dequantize_int4_gptq(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int = 128
):
    M, K = x.shape
    _, N = qweight.shape
    
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul4_kernel[grid](
        x, qweight, c,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1), qzeros.stride(2),
        GROUP_SIZE=group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c

# Example usage
x = torch.randn(512, 1024, device='cuda', dtype=torch.float16)
qweight = torch.randint(0, 255, (128, 4096), device='cuda', dtype=torch.int32)
scales = torch.randn(8, 4096, device='cuda', dtype=torch.float16)
qzeros = torch.randint(0, 255, (8, 4096, 16), device='cuda', dtype=torch.int32)

output = matmul_dequantize_int4_gptq(x, qweight, scales, qzeros, group_size=128)
