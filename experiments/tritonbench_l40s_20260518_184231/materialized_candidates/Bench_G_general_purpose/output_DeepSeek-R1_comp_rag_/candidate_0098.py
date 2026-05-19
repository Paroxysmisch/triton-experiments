import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, b_scale_ptr, b_zp_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_quant_k = offs_k // 8
    b_sub_k = offs_k % 8
    
    b_ptrs = b_ptr + b_quant_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b_packed = tl.load(b_ptrs, mask=(b_quant_k[:, None] < (K//8)) & (offs_n[None, :] < N), other=0)
    
    shift = (7 - b_sub_k[:, None]) * 4
    int4_val = (b_packed >> shift) & 0xF
    
    group_n = offs_n // GROUP_SIZE
    scale_ptrs = b_scale_ptr + offs_k[:, None] * stride_bsk + group_n[None, :] * stride_bsn
    scale = tl.load(scale_ptrs, mask=(offs_k[:, None] < K) & (group_n[None, :] < (N // GROUP_SIZE)), other=1.0)
    
    zp_quant_k = offs_k // 8
    zp_ptrs = b_zp_ptr + zp_quant_k[:, None] * stride_bzpk + group_n[None, :] * stride_bzpn
    zp_packed = tl.load(zp_ptrs, mask=(zp_quant_k[:, None] < (K//8)) & (group_n[None, :] < (N // GROUP_SIZE)), other=0)
    zp_val = (zp_packed >> shift) & 0xF
    
    b_dequant = (int4_val - zp_val) * scale
    
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    accumulator = tl.dot(a, b_dequant)
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul_dequantize_int4(a, b, b_scale, b_zp, group_size=128):
    M, K = a.shape
    Kw, N = b.shape
    assert Kw * 8 == K, "Quantized K dimension mismatch"
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    matmul_kernel[grid](
        a, b, b_scale, b_zp, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zp.stride(0), b_zp.stride(1),
        c.stride(0), c.stride(1),
        GROUP_SIZE=group_size
    )
    return c

def quantize_int4(weight, group_size=128):
    assert weight.dim() == 2, "Weight must be 2D"
    K, N = weight.shape
    assert N % group_size == 0, "N must be divisible by group_size"
    
    weight_flat = weight.view(K, N // group_size, group_size)
    max_vals = weight_flat.max(dim=-1, keepdim=True).values
    min_vals = weight_flat.min(dim=-1, keepdim=True).values
    
    scale = (max_vals - min_vals) / 15.0
    zp = (-min_vals / scale).round().clamp(0, 15)
    
    zp = zp.expand(-1, -1, group_size).contiguous().view(K, N)
    scale = scale.expand(-1, -1, group_size).contiguous().view(K, N)
    
    weight_int = ((weight - min_vals.expand_as(weight_flat)) / scale.expand_as(weight_flat)).round().clamp(0, 15)
    weight_int = weight_int.to(torch.int32)
    
    packed = torch.zeros((K // 8, N), dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed += weight_int[i::8] << (4 * (7 - i))
    
    scale = scale[:, ::group_size].contiguous()
    zp = zp[:, ::group_size].contiguous()
    return packed, scale, zp

def unpack_int4(packed, scale, zp, group_size=128):
    K_packed, N = packed.shape
    K = K_packed * 8
    
    unpacked = torch.zeros((K, N), dtype=torch.float32, device=packed.device)
    for i in range(8):
        val = (packed >> (4 * (7 - i))) & 0xF
        zp_tile = zp[i::8]
        scale_tile = scale[i::8]
        unpacked[i::8] = (val - zp_tile) * scale_tile
    return unpacked
