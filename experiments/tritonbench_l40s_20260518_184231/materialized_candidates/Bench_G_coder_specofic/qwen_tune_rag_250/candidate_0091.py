weights.

import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    # Pointers to matrices
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    # Matrix dimensions
    K, N, group_size,
    stride_bk, stride_bn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    stride_fpbk, stride_fpbn,
    # Meta-parameters
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Dequantize quantized int4 weights to fp32"""
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # Create offsets for memory access
    offsets_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Dequantize int4 weights using scales and zero points
    bscales = tl.load(b_scale_ptr + offsets_k * stride_bsk, mask=offsets_k < K, other=0.0).to(tl.float32)
    bzps = tl.load(b_zp_ptr + (offsets_k // group_size) * stride_bzpk, mask=offsets_k < K, other=0.0).to(tl.float32)
    b_4bit = tl.load(b_ptr + (offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn),
                    mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N), other=0).to(tl.int8)
    # Convert 4-bit integers to floating-point
    b_4bit = (b_4bit & 0x0F) | ((b_4bit & 0xF0) >> 4)
    bzps = (bzps + 1.0) * bscales
    fp_b = (b_4bit - bzps[None, :]) * bscales[None, :]
    # Store the result
    if SPLIT_K == 1:
        tl.store(fpb_ptr + (offsets_k[:, None] * stride_fpbk + offsets_n[None, :] * stride_fpbn),
                 fp_b, mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N))
    else:
        tl.atomic_add(fpb_ptr + (offsets_k[:, None] * stride_fpbk + offsets_n[None, :] * stride_fpbn),
                      fp_b, mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N))


@torch.no_grad()
def dequantize_int4(b, b_scale, b_zero_point):
    """Dequantize int4 weights to fp32"""
    K, N = b.shape
    b_scale = b_scale.squeeze()
    b_zero_point = b_zero_point.squeeze()
    group_size = 8
    fp_b = torch.empty((K, N), device=b.device, dtype=torch.float16)
    # Ensure inputs are on the correct device
    assert b_scale.device == b.device
    assert b_zero_point.device == b.device
    # Define the grid for the kernel launch
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    # Launch the Triton kernel
    dequantize_kernel[grid](
        b, b_scale, b_zero_point, fp_b,
        K, N, group_size,
        b.stride(0), b.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        b_zero_point.stride(0), b_zero_point.stride(1),
        fp_b.stride(0), fp_b.stride(1),
    )
    return fp_b
