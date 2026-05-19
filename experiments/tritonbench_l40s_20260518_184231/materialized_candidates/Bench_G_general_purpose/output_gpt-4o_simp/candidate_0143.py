import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_fwd_kernel(a_ptr, b_ptr, out_ptr,
                          M, N, K,  # dimensions of the matrices
                          stride_am, stride_ak,  # strides for matrix A
                          stride_bn, stride_bk,  # strides for matrix B
                          stride_om, stride_on,  # strides for output matrix
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          causal: tl.constexpr):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define the start of the block
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    # Create pointers for the current block of A and B
    a_block_ptr = a_ptr + block_start_m * stride_am
    b_block_ptr = b_ptr + block_start_n * stride_bn

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, K, BLOCK_K):
        # Load blocks of A and B
        a_block = tl.load(a_block_ptr + k * stride_ak, mask=(block_start_m + tl.arange(0, BLOCK_M)[:, None] < M))
        b_block = tl.load(b_block_ptr + k * stride_bk, mask=(block_start_n + tl.arange(0, BLOCK_N)[None, :] < N))

        # Update the accumulator
        acc += tl.dot(a_block, b_block)

    # Apply causal masking if required
    if causal:
        mask = tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N)[None, :]
        acc = tl.where(mask, acc, 0)

    # Store the result
    out_ptr = out_ptr + block_start_m * stride_om + block_start_n * stride_on
    tl.store(out_ptr, acc, mask=(block_start_m + tl.arange(0, BLOCK_M)[:, None] < M) & (block_start_n + tl.arange(0, BLOCK_N)[None, :] < N))

import torch

def _bmm_chunk_fwd(a, b, chunk_size, causal=False):
    # Dimensions
    B, M, K = a.shape
    _, _, N = b.shape

    # Allocate output
    out = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    # Strides
    stride_am, stride_ak = a.stride(-2), a.stride(-1)
    stride_bn, stride_bk = b.stride(-1), b.stride(-2)
    stride_om, stride_on = out.stride(-2), out.stride(-1)

    # Launch kernel
    grid = (triton.cdiv(M, chunk_size), triton.cdiv(N, chunk_size), B)
    _bmm_chunk_fwd_kernel[grid](
        a, b, out,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_om, stride_on,
        BLOCK_M=chunk_size, BLOCK_N=chunk_size, BLOCK_K=chunk_size,
        causal=causal
    )

    return out
