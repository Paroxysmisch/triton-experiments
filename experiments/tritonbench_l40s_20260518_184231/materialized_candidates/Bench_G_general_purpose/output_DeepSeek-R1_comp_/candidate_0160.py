import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, scales_ptr, zeros_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_scales_g, stride_scales_k,
    stride_zeros_g, stride_zeros_k,
    stride_cm, stride_cn,
    M, N, K,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + (offs_n // 8)[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K
        a_mask = (offs_m[:, None] < M) & (k_offs + offs_k[None, :] < K)
        a = tl.load(a_ptrs + k_offs, mask=a_mask, other=0.0)

        b_mask = (k_offs + offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_packed = tl.load(b_ptrs + k_offs * stride_bk, mask=b_mask, other=0)

        b = (b_packed >> (tl.arange(0, 8) * 4).to(tl.int32)[None, None, :]) & 0xF
        b = tl.view(b, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        group_idx = pid_n * BLOCK_SIZE_N // group_size
        scale_ptrs = scales_ptr + group_idx * stride_scales_g + (k_offs + offs_k) * stride_scales_k
        scales = tl.load(scale_ptrs, mask=(k_offs + offs_k) < K, other=0.0)

        zero_ptrs = zeros_ptr + group_idx * stride_zeros_g + (k_offs + offs_k) * stride_zeros_k
        zeros = tl.load(zero_ptrs, mask=(k_offs + offs_k) < K, other=0)

        b = (b - zeros[:, None]).to(tl.float32) * scales[:, None]
        acc += tl.dot(a.to(tl.float32), b, allow_tf32=False)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)

def matmul_dequantize_int4_gptq(a, b, scales, zeros, group_size, out=None):
    M, K = a.shape
    _, N_packed = b.shape
    N = N_packed * 8
    assert scales.shape == (N // group_size, K), "Scales shape mismatch"
    assert zeros.shape == (N // group_size, K), "Zeros shape mismatch"

    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (M, N), "Out shape mismatch"

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    matmul4_kernel[grid](
        a, b, scales, zeros, out,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        out.stride(0), out.stride(1),
        M, N, K,
        group_size=group_size,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
    )
    return out

def quantize_int4(weight, group_size=64):
    N, K = weight.shape
    assert N % group_size == 0, "N must be divisible by group_size"
    num_groups = N // group_size
    
    weight = weight.reshape(num_groups, group_size, K)
    scales = torch.zeros((num_groups, K), dtype=torch.float16, device=weight.device)
    zeros = torch.zeros((num_groups, K), dtype=torch.uint8, device=weight.device)
    packed = torch.zeros((K, N // 8), dtype=torch.int32, device=weight.device)

    for g in range(num_groups):
        group = weight[g]
        max_vals = group.max(dim=0).values
        min_vals = group.min(dim=0).values
        scales[g] = (max_vals - min_vals) / 15.0
        zeros[g] = torch.clamp((-min_vals / scales[g]).round(), 0, 15).to(torch.uint8)
        quantized = torch.clamp(((group - min_vals) / scales[g]).round(), 0, 15).to(torch.uint8)
        
        packed_group = torch.zeros((K, group_size // 8), dtype=torch.int32, device=weight.device)
        for i in range(8):
            packed_group |= (quantized[:, :, i::8] << (4 * (7 - i))).to(torch.int32)
        packed[:, g*(group_size//8):(g+1)*(group_size//8)] = packed_group.transpose(0, 1)
    
    return packed.contiguous(), scales.contiguous(), zeros.contiguous()
