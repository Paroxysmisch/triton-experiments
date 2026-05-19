import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, scales_ptr, zeros_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (offs_k[None, :] * stride_ak)
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    num_groups = BLOCK_SIZE_K // GROUP_SIZE
    for g in range(num_groups):
        g_offs_k = g * GROUP_SIZE
        group_idx = (pid_k * BLOCK_SIZE_K + g_offs_k) // GROUP_SIZE
        
        b_ptrs = b_ptr + offs_n[None, :] * stride_bn + (group_idx + tl.arange(0, BLOCK_SIZE_N)[None, :] * (K // GROUP_SIZE))
        scales_ptrs = scales_ptr + offs_n[None, :] * stride_scales_n + group_idx
        zeros_ptrs = zeros_ptr + offs_n[None, :] * stride_zeros_n + group_idx

        packed = tl.load(b_ptrs)
        scale = tl.load(scales_ptrs)
        zero = tl.load(zeros_ptrs)

        for i in tl.static_range(GROUP_SIZE):
            shift = i * 4
            elem = (packed >> shift) & 0xF
            elem = elem.to(tl.int8, bitcast=True)  # Sign extend
            elem = elem.to(tl.float32) * scale + zero
            k_idx = g_offs_k + i
            a_col = tl.load(a_ptr + offs_m[:, None] * stride_am + (pid_k * BLOCK_SIZE_K + k_idx) * stride_ak, 
                          mask=(offs_m[:, None] < M) & ((pid_k * BLOCK_SIZE_K + k_idx) < K), other=0.0)
            acc += tl.dot(a_col, elem, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul_dequantize_int4_s2(a: torch.Tensor, b_q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
    M, K = a.shape
    _, N = scales.shape
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        1
    )
    
    matmul_kernel[grid](
        a, b_q, scales, zeros, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_q.stride(0), b_q.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        c.stride(0), c.stride(1),
        GROUP_SIZE=8
    )
    return c

def quantize_int4(w: torch.Tensor):
    orig_shape = w.shape
    w = w.reshape(-1, 8)
    
    scale = (w.max(dim=1).values - w.min(dim=1).values) / 15
    zero = w.min(dim=1).values
    
    w_quant = torch.clamp(torch.round((w - zero[:, None]) / scale[:, None]), 0, 15).to(torch.int32)
    
    packed = torch.zeros((w_quant.shape[0], 1), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w_quant[:, i] & 0xF).to(torch.int32) << (i * 4)
    
    return packed.reshape(orig_shape[0], orig_shape[1]//8), scale.reshape(orig_shape[0], -1), zero.reshape(orig_shape[0], -1)

def unpack_int4(w_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
    w = torch.zeros((w_packed.shape[0], w_packed.shape[1] * 8), dtype=torch.float32, device=w_packed.device)
    for i in range(8):
        w[:, i::8] = ((w_packed >> (i * 4)) & 0xF).float() * scales + zeros
    return w
