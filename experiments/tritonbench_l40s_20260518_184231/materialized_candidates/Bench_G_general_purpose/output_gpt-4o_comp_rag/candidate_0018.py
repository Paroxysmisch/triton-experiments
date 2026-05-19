import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(
    q_ptr, q_int8_ptr, q_scale_ptr,
    M, N, BLKQ: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLKQ + tl.arange(0, BLKQ)
    offs_n = tl.arange(0, N)
    q_ptrs = q_ptr + offs_m[:, None] * N + offs_n[None, :]
    q_int8_ptrs = q_int8_ptr + offs_m[:, None] * N + offs_n[None, :]
    q_scale_ptrs = q_scale_ptr + offs_m

    q_block = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    q_max = tl.max(tl.abs(q_block), axis=1)
    q_scale = q_max / 127.0
    q_block_quantized = tl.libdevice.round(q_block / q_scale[:, None]).to(tl.int8)

    tl.store(q_int8_ptrs, q_block_quantized, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    tl.store(q_scale_ptrs, q_scale, mask=offs_m < M)

@triton.jit
def k_kernel_per_block_int8(
    k_ptr, k_int8_ptr, k_scale_ptr,
    M, N, BLKK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLKK + tl.arange(0, BLKK)
    offs_n = tl.arange(0, N)
    k_ptrs = k_ptr + offs_m[:, None] * N + offs_n[None, :]
    k_int8_ptrs = k_int8_ptr + offs_m[:, None] * N + offs_n[None, :]
    k_scale_ptrs = k_scale_ptr + offs_m

    k_block = tl.load(k_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    k_max = tl.max(tl.abs(k_block), axis=1)
    k_scale = k_max / 127.0
    k_block_quantized = tl.libdevice.round(k_block / k_scale[:, None]).to(tl.int8)

    tl.store(k_int8_ptrs, k_block_quantized, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    tl.store(k_scale_ptrs, k_scale, mask=offs_m < M)

def per_block_int8(q, k, BLKQ, BLKK):
    M, N = q.shape
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    k_int8 = torch.empty_like(k, dtype=torch.int8)
    q_scale = torch.empty(M, device=q.device, dtype=torch.float32)
    k_scale = torch.empty(M, device=k.device, dtype=torch.float32)

    grid_q = lambda META: (triton.cdiv(M, BLKQ),)
    grid_k = lambda META: (triton.cdiv(M, BLKK),)

    q_kernel_per_block_int8[grid_q](
        q, q_int8, q_scale,
        M, N, BLKQ
    )

    k_kernel_per_block_int8[grid_k](
        k, k_int8, k_scale,
        M, N, BLKK
    )

    return q_int8, k_int8, q_scale, k_scale

# Example usage
q = torch.randn(1024, 512, device='cuda', dtype=torch.float32)
k = torch.randn(1024, 512, device='cuda', dtype=torch.float32)
BLKQ = 128
BLKK = 128

q_int8, k_int8, q_scale, k_scale = per_block_int8(q, k, BLKQ, BLKK)
