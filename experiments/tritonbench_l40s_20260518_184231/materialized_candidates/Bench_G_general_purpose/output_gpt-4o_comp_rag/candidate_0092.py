import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[:, None] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if SPLIT_K > 1:
        c_ptrs += pid_k * stride_cm * BLOCK_SIZE_M * stride_cn * BLOCK_SIZE_N
        tl.atomic_add(c_ptrs, acc)
    else:
        tl.store(c_ptrs, acc)

def quantize_int4(weights):
    scales = torch.max(torch.abs(weights), dim=0, keepdim=True).values / 7.5
    zero_points = torch.zeros_like(scales, dtype=torch.int32)
    quantized_weights = torch.round(weights / scales).to(torch.int32)
    packed_weights = torch.zeros((weights.shape[0] // 8, weights.shape[1]), dtype=torch.int32)

    for i in range(8):
        packed_weights |= ((quantized_weights[i::8] & 0xF) << (i * 4))

    return packed_weights, scales, zero_points

def unpack_int4(packed_weights, scales, zero_points):
    unpacked_weights = torch.zeros((packed_weights.shape[0] * 8, packed_weights.shape[1]), dtype=torch.float32)
    for i in range(8):
        int4_values = (packed_weights >> (i * 4)) & 0xF
        unpacked_weights[i::8] = (int4_values - zero_points) * scales

    return unpacked_weights

def matmul_dequantize_int4_s2(a, b, b_scale, b_zero_point, group_size=128, out=None):
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    Kw, N = b.shape
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out

    fp_b = dequantize_int4(b, b_scale, b_zero_point, a.device, a.dtype, group_size)
    torch.mm(a, fp_b, out=c)
    fp_b = None
    return c

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
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_offs = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    bzp_offs = offs_k[:, None] * stride_bzpk + (offs_n // group_size)[None, :] * stride_bzpn
    n_mask = offs_n[None, :] < N
    k_mask = offs_k[:, None] < K
    mask = n_mask & k_mask
    int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
    for i in range(8):
        int4_b = ((int32_b << (28 - i * 4) >> 28) + 16) & 15
        int4_zp = ((zp_b << (28 - i * 4) >> 28) + 16) & 15
        bs_offs = (offs_k * 8 + i)[:, None] * stride_bsk + (offs_n // group_size)[None, :] * stride_bsn
        fpb_offs = (offs_k * 8 + i)[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn
        k8_mask = (offs_k * 8 + i)[:, None] < K * 8
        scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask & k8_mask, other=0.0)
        fp_weight = (int4_b - int4_zp) * scale_b
        tl.store(fpb_ptr + fpb_offs, fp_weight, mask=n_mask & k8_mask)

def dequantize_int4(b, b_scale, b_zero_point, device, dtype, group_size):
    Kw, N = b.shape
    fp_b = torch.empty((b_scale.shape[0], b.shape[1]), device=device, dtype=dtype)
    grid = lambda META: (
        triton.cdiv(Kw, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N']), 
    )
    dequantize_kernel[grid](
        b, b_scale, b_zero_point, fp_b,
        Kw, N, group_size,
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zero_point.stride(0), b_zero_point.stride(1),
        fp_b.stride(0), fp_b.stride(1)
    )
    return fp_b

def matmul_dequantize_int4(a, b, b_scale, b_zero_point, group_size=128, out=None):
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    Kw, N = b.shape
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
    fp_b = dequantize_int4(b, b_scale, b_zero_point, a.device, a.dtype, group_size)
    torch.mm(a, fp_b, out=c)
    fp_b = None
    return c
