import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_om, stride_on,
    chunk_size,
    causal, seq_index,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Compute program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_block = num_pid_m * num_pid_n
    block_id = pid // num_pid_in_block
    pid_m = (pid % num_pid_in_block) // num_pid_n
    pid_n = (pid % num_pid_in_block) % num_pid_n

    # Compute block bounds
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = block_id * chunk_size

    # Initialize output block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Load and compute dot products
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute bounds for current chunk
        block_start_k_chunk = block_start_k + k
        block_end_k_chunk = min(block_start_k_chunk + BLOCK_SIZE_K, K)

        # Load sub-matrices
        a = tl.load(a_ptr + block_start_m * stride_am + block_start_k_chunk * stride_ak, 
                    mask=block_start_k_chunk + tl.arange(0, BLOCK_SIZE_K) < K, 
                    other=0.0)
        b = tl.load(b_ptr + block_start_k_chunk * stride_bk + block_start_n * stride_bn, 
                    mask=block_start_k_chunk + tl.arange(0, BLOCK_SIZE_K) < K, 
                    other=0.0)

        # Compute dot product
        acc += tl.dot(a, b)

        # Apply causal masking if needed
        if causal:
            causal_mask = block_start_m + tl.arange(0, BLOCK_SIZE_M) >= block_start_n + tl.arange(0, BLOCK_SIZE_N)
            acc = tl.where(causal_mask, acc, float('-inf'))

        # Apply sequence indexing if needed
        if seq_index:
            seq_mask = block_start_m + tl.arange(0, BLOCK_SIZE_M) == block_start_n + tl.arange(0, BLOCK_SIZE_N)
            acc = tl.where(seq_mask, acc, 0.0)

    # Store result
    tl.store(out_ptr + block_start_m * stride_om + block_start_n * stride_on, acc)

import torch

def _bmm_chunk_fwd(a, b, out, chunk_size, causal=False, seq_index=False, block_size_m=16, block_size_n=16, block_size_k=16):
    # Get dimensions
    B, M, K = a.shape
    B, K, N = b.shape

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(K, chunk_size),
    )
    _bmm_chunk_fwd_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(1), a.stride(2), b.stride(1), b.stride(2),
        out.stride(1), out.stride(2),
        chunk_size,
        causal, seq_index,
        BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n, BLOCK_SIZE_K=block_size_k
    )

    return out
