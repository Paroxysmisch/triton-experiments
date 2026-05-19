import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize,
    NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 2) * stride_bk + offs_n[None, :] * stride_bn
    
    shifter = (offs_k % 2) * 4
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0)
        
        b_values = (b_packed >> shifter[:, None]) & 0xF
        b_values = b_values.to(tl.float16)
        
        if NO_GROUPS:
            g = 0
        else:
            g = (k * BLOCK_SIZE_K + offs_k) // groupsize
        
        scale_ptrs = scales_ptr + g[:, None] * stride_scales_g + offs_n[None, :] * stride_scales_n
        zp_ptrs = zeros_ptr + g[:, None] * stride_zeros_g + offs_n[None, :] * stride_zeros_n
        
        scales = tl.load(scale_ptrs, mask=b_mask, other=1.0)
        zeros = tl.load(zp_ptrs, mask=b_mask, other=0.0)
        
        b = (b_values - zeros) * scales
        acc += tl.dot(a, b, allow_tf32=True)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 128}, num_warps=4),
    ],
    key=['K', 'N']
)
@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    off_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_k = off_k < K
    mask_n = off_n < N
    mask = mask_k[:, None] & mask_n[None, :]
    
    packed_k = off_k // 2
    shifter = (off_k % 2) * 4
    
    b_ptrs = b_ptr + packed_k[:, None] * stride_bk + off_n[None, :] * stride_bn
    b_packed = tl.load(b_ptrs, mask=mask & (packed_k[:, None] < (K // 2)), other=0)
    b_values = (b_packed >> shifter[:, None]) & 0xF
    
    g = off_k // group_size
    scale_ptrs = b_scale_ptr + g[:, None] * stride_bsk + off_n[None, :] * stride_bsn
    zp_ptrs = b_zp_ptr + g[:, None] * stride_bzpk + off_n[None, :] * stride_bzpn
    
    scales = tl.load(scale_ptrs, mask=mask, other=1.0)
    zeros = tl.load(zp_ptrs, mask=mask, other=0.0)
    
    dequantized = (b_values.to(tl.float16) - zeros) * scales
    fpb_ptrs = fpb_ptr + off_k[:, None] * stride_fpbk + off_n[None, :] * stride_fpbn
    tl.store(fpb_ptrs, dequantized, mask=mask)

def dequantize_int4(b: torch.Tensor, b_scale: torch.Tensor, b_zp: torch.Tensor, group_size: int):
    K, N = b.shape
    fpb = torch.empty((K, N), device=b.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(K, meta['BLOCK_SIZE_K']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    dequantize_kernel[grid](b, b_scale, b_zp, fpb, K, N, group_size,
                            b.stride(0), b.stride(1),
                            b_scale.stride(0), b_scale.stride(1),
                            b_zp.stride(0), b_zp.stride(1),
                            fpb.stride(0), fpb.stride(1))
    return fpb

def matmul_dequantize_int4_s1(a: torch.Tensor, b: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, groupsize: int):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    NO_GROUPS = groupsize == K
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    matmul4_kernel[grid](a, b, c, scales, zeros, M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        scales.stride(0), scales.stride(1),
                        zeros.stride(0), zeros.stride(1),
                        groupsize, NO_GROUPS)
    return c

def quantize_int4(weight: torch.Tensor, group_size: int = 128):
    org_weight = weight.clone()
    weight = weight.view(-1, group_size)
    scale = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    zp = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    
    for i in range(weight.shape[0]):
        w = weight[i]
        max_val = w.max()
        min_val = w.min()
        scale[i] = (max_val - min_val) / 15
        zp[i] = torch.round(min_val / scale[i])
        quantized = torch.clamp(torch.round(w / scale[i]) - zp[i], 0, 15)
        weight[i] = quantized
    
    weight = weight.view(org_weight.shape)
    int_weight = torch.zeros(weight.numel() // 2, device=weight.device, dtype=torch.int8)
    
    for i in range(weight.numel() // 2):
        high = weight[2*i].int() << 4
        low = weight[2*i+1].int() & 0x0F
        int_weight[i] = (high | low).to(torch.int8)
    
    return int_weight, scale.view(-1, 1), zp.view(-1, 1)
