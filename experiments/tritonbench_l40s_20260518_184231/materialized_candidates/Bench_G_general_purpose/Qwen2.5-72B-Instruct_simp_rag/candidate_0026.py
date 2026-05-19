import torch
import triton
import triton.language as tl

# Triton kernel for converting query matrix to int8
@triton.jit
def q_kernel_per_block_int8(
    q_ptr, q_int8_ptr, q_scale_ptr,
    M, N,
    stride_qm, stride_qn,
    stride_qint8m, stride_qint8n,
    stride_qscalem,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start_m = pid * BLOCK_SIZE_M
    block_start_n = pid * BLOCK_SIZE_N

    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offsets_m < M
    mask_n = offsets_n < N

    q_ptrs = q_ptr + offsets_m[:, None] * stride_qm + offsets_n[None, :] * stride_qn
    q_values = tl.load(q_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Compute the scaling factor for this block
    max_val = tl.max(q_values, axis=None)
    min_val = tl.min(q_values, axis=None)
    scale = (max_val - min_val) / 255.0
    q_scale = tl.full((1,), scale, dtype=tl.float32)
    q_scale_ptr += block_start_m * stride_qscalem
    tl.store(q_scale_ptr, q_scale)

    # Quantize the values
    q_int8_values = tl.round((q_values - min_val) / scale)
    q_int8_values = tl.where(q_int8_values < 0, 0, q_int8_values)
    q_int8_values = tl.where(q_int8_values > 255, 255, q_int8_values)
    q_int8_values = q_int8_values.to(tl.int8)

    q_int8_ptrs = q_int8_ptr + offsets_m[:, None] * stride_qint8m + offsets_n[None, :] * stride_qint8n
    tl.store(q_int8_ptrs, q_int8_values, mask=mask_m[:, None] & mask_n[None, :])

# Triton kernel for converting key matrix to int8
@triton.jit
def k_kernel_per_block_int8(
    k_ptr, k_int8_ptr, k_scale_ptr,
    M, N,
    stride_km, stride_kn,
    stride_kint8m, stride_kint8n,
    stride_kscalem,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start_m = pid * BLOCK_SIZE_M
    block_start_n = pid * BLOCK_SIZE_N

    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offsets_m < M
    mask_n = offsets_n < N

    k_ptrs = k_ptr + offsets_m[:, None] * stride_km + offsets_n[None, :] * stride_kn
    k_values = tl.load(k_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Compute the scaling factor for this block
    max_val = tl.max(k_values, axis=None)
    min_val = tl.min(k_values, axis=None)
    scale = (max_val - min_val) / 255.0
    k_scale = tl.full((1,), scale, dtype=tl.float32)
    k_scale_ptr += block_start_m * stride_kscalem
    tl.store(k_scale_ptr, k_scale)

    # Quantize the values
    k_int8_values = tl.round((k_values - min_val) / scale)
    k_int8_values = tl.where(k_int8_values < 0, 0, k_int8_values)
    k_int8_values = tl.where(k_int8_values > 255, 255, k_int8_values)
    k_int8_values = k_int8_values.to(tl.int8)

    k_int8_ptrs = k_int8_ptr + offsets_m[:, None] * stride_kint8m + offsets_n[None, :] * stride_kint8n
    tl.store(k_int8_ptrs, k_int8_values, mask=mask_m[:, None] & mask_n[None, :])

# Wrapper function to manage the conversion process
def per_block_int8(q, k, BLKQ, BLKK):
    M, N = q.shape
    q_int8 = torch.empty((M, N), dtype=torch.int8, device=q.device)
    q_scale = torch.empty((M // BLKQ, 1), dtype=torch.float32, device=q.device)
    k_int8 = torch.empty((M, N), dtype=torch.int8, device=k.device)
    k_scale = torch.empty((M // BLKK, 1), dtype=torch.float32, device=k.device)

    grid_q = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),)
    grid_k = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),)

    q_kernel_per_block_int8[grid_q](
        q, q_int8, q_scale,
        M, N,
        q.stride(0), q.stride(1),
        q_int8.stride(0), q_int8.stride(1),
        q_scale.stride(0),
        BLOCK_SIZE_M=BLKQ, BLOCK_SIZE_N=BLKQ
    )

    k_kernel_per_block_int8[grid_k](
        k, k_int8, k_scale,
        M, N,
        k.stride(0), k.stride(1),
        k_int8.stride(0), k_int8.stride(1),
        k_scale.stride(0),
        BLOCK_SIZE_M=BLKK, BLOCK_SIZE_N=BLKK
    )

    return q_int8, q_scale, k_int8, k_scale

# Example usage
q = torch.randn((1024, 1024), device='cuda')
k = torch.randn((1024, 1024), device='cuda')
BLKQ = 128
BLKK = 128

q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, BLKQ, BLKK)
