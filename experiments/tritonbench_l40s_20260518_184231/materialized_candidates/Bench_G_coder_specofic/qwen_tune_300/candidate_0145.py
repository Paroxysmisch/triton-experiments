import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    b,
    b_scale,
    fpb,
    K,
    N,
    stride_b_k,
    stride_b_n,
    stride_b_scale_k,
    stride_b_scale_n,
    stride_fpb_k,
    stride_fpb_n,
):
    # Map program ids `pid_n` and `pid_k` to the block of C it should compute.
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    # 'block_n' is the start of the block of N that the program will go through.
    block_n = pid_n * BLOCK_SIZE_N
    # 'block_k' is the start of the block of K that the program will go through.
    block_k = pid_k * BLOCK_SIZE_K
    # The first few blocks are loaded from the start of the file.
    offset_b = block_k * stride_b_k + (block_n) * stride_b_n
    offset_b_scale = block_k * stride_b_scale_k + (block_n) * stride_b_scale_n
    offs_k = block_k + tl.arange(0, BLOCK_SIZE_K)
    offs_n = block_n + tl.arange(0, BLOCK_SIZE_N)
    # Load b and b_scale segments. These are fixed across the block computation.
    b = tl.load(
        b + offset_b,
        mask=(offs_k < K)[:, None] & (offs_n < N)[None, :],
        other=0.0,
    )
    b_scale = tl.load(
        b_scale + offset_b_scale,
        mask=(offs_k < K)[:, None] & (offs_n < N)[None, :],
        other=0.0,
    )
    # Compute segments of C. These are fixed across the block computation.
    c_segment = tl.dot(b, b_scale)
    # Write back the result.
    offset_fpb = block_k * stride_fpb_k + (block_n) * stride_fpb_n
    tl.store(
        fpb + offset_fpb,
        c_segment,
        mask=(offs_k < K)[:, None] & (offs_n < N)[None, :],
    )


def matmul_dequantize_int8(a, b, b_scale):
    # Check constraints.
    assert a.shape[1] == b.shape[0] and a.shape[2] == b.shape[1]
    batch, K, N = a.shape
    _, _, scale_K, scale_N = b_scale.shape
    assert K == scale_K and N == scale_N
    # Allocates output.
    c = torch.empty((batch, K, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block computes a block of the C matrix.
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
        triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )
    dequantize_kernel[grid](
        b,
        b_scale,
        c,
        K,
        N,
        b.stride(1),
        b.stride(2),
        b_scale.stride(1),
        b_scale.stride(2),
        c.stride(1),
        c.stride(2),
    )
    # a @ b
    return torch.mm(a, c)
