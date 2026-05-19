import triton
import triton.language as tl
import torch

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr,
    stride_a_batch, stride_a_row, stride_a_col,
    stride_b_batch, stride_b_row, stride_b_col,
    stride_out_batch, stride_out_row, stride_out_col,
    M, N, K,
    chunk_size, causal, seq_idx,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (M // BLOCK_M)
    row_id = pid % (M // BLOCK_M)

    offs_m = row_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + batch_id * stride_a_batch + offs_m[:, None] * stride_a_row + offs_k[None, :] * stride_a_col
    b_ptrs = b_ptr + batch_id * stride_b_batch + offs_k[:, None] * stride_b_row + offs_n[None, :] * stride_b_col
    out_ptrs = out_ptr + batch_id * stride_out_batch + offs_m[:, None] * stride_out_row + offs_n[None, :] * stride_out_col

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_chunk in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_chunk, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_chunk, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_a_col
        b_ptrs += BLOCK_K * stride_b_row

    if causal:
        mask = tl.arange(0, BLOCK_N)[:, None] >= (seq_idx + offs_m[:, None])
        acc = tl.where(mask, acc, 0.0)

    tl.store(out_ptrs, acc)

def _bmm_chunk_fwd(a, b, chunk_size, causal=False, seq_idx=None):
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions for matrix multiplication"

    batch_size, M, K = a.shape
    _, _, N = b.shape

    out = torch.empty((batch_size, M, N), device=a.device, dtype=a.dtype)

    grid = (batch_size * (M + chunk_size - 1) // chunk_size,)

    triton._bmm_chunk_fwd_kernel[grid](
        a, b, out,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        M, N, K,
        chunk_size, causal, seq_idx,
        BLOCK_M=chunk_size, BLOCK_N=chunk_size, BLOCK_K=32  # Example block sizes
    )

    return out
