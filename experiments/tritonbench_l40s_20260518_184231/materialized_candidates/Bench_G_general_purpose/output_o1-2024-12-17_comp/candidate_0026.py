import triton
import triton.language as tl
import torch

@triton.jit
def q_kernel_per_block_int8(
    q_ptr, q_scale_ptr, q_int8_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    M, N,
    stride_qM, stride_qN,
    stride_scaleM,
    stride_int8M, stride_int8N
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Create 2D index
    q_idx = offs_m[:, None] * stride_qM + offs_n[None, :] * stride_qN

    # Load block
    q_block = tl.where(mask_m[:, None] & mask_n[None, :],
                       tl.load(q_ptr + q_idx, mask=mask_m[:, None] & mask_n[None, :]),
                       0.0)

    # Find block max abs
    abs_block = tl.abs(q_block)
    block_max = tl.max(tl.max(abs_block, 1), 0)

    # Compute scale
    scale = 127.0 / (block_max + 1e-8)

    # Normalize and quantize
    q_quantized = tl.libdevice.round(q_block * scale)
    q_quantized = tl.clip(q_quantized, -127.0, 127.0)

    # Store scale (one per row-block)
    scale_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    scale_idx = scale_offs * stride_scaleM
    # All rows in the block share the same block-level max -> broadcast first row's scale
    scale_val = tl.where(mask_m, scale, 0.0)

    tl.store(q_scale_ptr + scale_idx, scale_val, mask=mask_m)

    # Store quantized block
    int8_idx = offs_m[:, None] * stride_int8M + offs_n[None, :] * stride_int8N
    tl.store(q_int8_ptr + int8_idx, q_quantized.to(tl.int8),
             mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def k_kernel_per_block_int8(
    k_ptr, k_scale_ptr, k_int8_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    M, N,
    stride_kM, stride_kN,
    stride_scaleM,
    stride_int8M, stride_int8N
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Create 2D index
    k_idx = offs_m[:, None] * stride_kM + offs_n[None, :] * stride_kN

    # Load block
    k_block = tl.where(mask_m[:, None] & mask_n[None, :],
                       tl.load(k_ptr + k_idx, mask=mask_m[:, None] & mask_n[None, :]),
                       0.0)

    # Find block max abs
    abs_block = tl.abs(k_block)
    block_max = tl.max(tl.max(abs_block, 1), 0)

    # Compute scale
    scale = 127.0 / (block_max + 1e-8)

    # Normalize and quantize
    k_quantized = tl.libdevice.round(k_block * scale)
    k_quantized = tl.clip(k_quantized, -127.0, 127.0)

    # Store scale (one per row-block)
    scale_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    scale_idx = scale_offs * stride_scaleM
    scale_val = tl.where(mask_m, scale, 0.0)

    tl.store(k_scale_ptr + scale_idx, scale_val, mask=mask_m)

    # Store quantized block
    int8_idx = offs_m[:, None] * stride_int8M + offs_n[None, :] * stride_int8N
    tl.store(k_int8_ptr + int8_idx, k_quantized.to(tl.int8),
             mask=mask_m[:, None] & mask_n[None, :])

def per_block_int8(q, k, BLKQ, BLKK):
    # Reshape q and k to 2D if they are 3D or 4D
    original_q_shape = q.shape
    original_k_shape = k.shape
    q_2d = q.reshape(-1, q.shape[-1])
    k_2d = k.reshape(-1, k.shape[-1])

    Mq, Nq = q_2d.shape
    Mk, Nk = k_2d.shape

    # Allocate output int8 tensors and scale tensors
    q_int8 = torch.empty_like(q_2d, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty_like(k_2d, dtype=torch.int8, device=k.device)
    q_scale = torch.empty(Mq, dtype=q.dtype, device=q.device)
    k_scale = torch.empty(Mk, dtype=k.dtype, device=k.device)

    grid_q = lambda meta: (
        (Mq + BLKQ - 1) // BLKQ,
        (Nq + BLKQ - 1) // BLKQ
    )
    grid_k = lambda meta: (
        (Mk + BLKK - 1) // BLKK,
        (Nk + BLKK - 1) // BLKK
    )

    q_kernel_per_block_int8[grid_q](
        q_2d, q_scale, q_int8,
        BLOCK_M=BLKQ, BLOCK_N=BLKQ,
        M=Mq, N=Nq,
        stride_qM=q_2d.stride(0), stride_qN=q_2d.stride(1),
        stride_scaleM=q_scale.stride(0),
        stride_int8M=q_int8.stride(0), stride_int8N=q_int8.stride(1)
    )

    k_kernel_per_block_int8[grid_k](
        k_2d, k_scale, k_int8,
        BLOCK_M=BLKK, BLOCK_N=BLKK,
        M=Mk, N=Nk,
        stride_kM=k_2d.stride(0), stride_kN=k_2d.stride(1),
        stride_scaleM=k_scale.stride(0),
        stride_int8M=k_int8.stride(0), stride_int8N=k_int8.stride(1)
    )

    # Reshape the int8 and scale back to original shapes
    q_int8 = q_int8.view(*original_q_shape).contiguous()
    k_int8 = k_int8.view(*original_k_shape).contiguous()
    q_scale = q_scale.view(-1).contiguous()
    k_scale = k_scale.view(-1).contiguous()

    return q_int8, k_int8, q_scale, k_scale
