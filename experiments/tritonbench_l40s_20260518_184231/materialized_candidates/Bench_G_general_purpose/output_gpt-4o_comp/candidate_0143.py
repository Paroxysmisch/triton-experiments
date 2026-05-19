import triton
import triton.language as tl
import torch

# Triton kernel for batched matrix multiplication with chunking, causal masking, and sequence indexing
@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr,
    a_batch_stride, a_head_stride, a_m_stride, a_k_stride,
    b_batch_stride, b_head_stride, b_k_stride, b_n_stride,
    out_batch_stride, out_head_stride, out_m_stride, out_n_stride,
    seq_idx_ptr, a_seq_stride, out_seq_stride,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr, HAS_SEQ_IDX: tl.constexpr
):
    pid = tl.program_id(0)
    # Calculate batch, head, and chunk indices
    batch_idx = pid // (head_dim * chunk_dim)
    head_idx = (pid // chunk_dim) % head_dim
    chunk_idx = pid % chunk_dim

    # Compute the starting position for each block
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the start of each sub-matrix
    a_ptrs = a_ptr + batch_idx * a_batch_stride + head_idx * a_head_stride + chunk_idx * BLOCK_SIZE_K * a_k_stride
    b_ptrs = b_ptr + batch_idx * b_batch_stride + head_idx * b_head_stride + chunk_idx * BLOCK_SIZE_K * b_k_stride

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, BLOCK_SIZE_K):
        a = tl.load(a_ptrs + offs_m[:, None] * a_m_stride + offs_k[None, :] * a_k_stride)
        b = tl.load(b_ptrs + offs_k[:, None] * b_k_stride + offs_n[None, :] * b_n_stride)
        acc += tl.dot(a, b)

    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + batch_idx * a_seq_stride + offs_m)
        seq_idx_out = tl.load(seq_idx_ptr + batch_idx * out_seq_stride + offs_n)
        mask = seq_idx[:, None] == seq_idx_out[None, :]
        acc = tl.where(mask, acc, 0.0)

    if IS_CAUSAL:
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        acc = tl.where(causal_mask, acc, 0.0)

    out_ptrs = out_ptr + batch_idx * out_batch_stride + head_idx * out_head_stride
    tl.store(out_ptrs + offs_m[:, None] * out_m_stride + offs_n[None, :] * out_n_stride, acc)

# Python wrapper function
def _bmm_chunk_fwd(a, b, out=None, seq_idx=None, causal=False):
    # Extract dimensions
    batch_size, head_dim, m_dim, k_dim = a.shape
    _, _, _, n_dim = b.shape

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Allocate output if not provided
    if out is None:
        out = torch.empty((batch_size, head_dim, m_dim, n_dim), device=a.device, dtype=a.dtype)

    # Determine grid size
    grid = (batch_size * head_dim * (m_dim // BLOCK_SIZE_M),)

    # Launch kernel
    _bmm_chunk_fwd_kernel[grid](
        a, b, out,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        seq_idx if seq_idx is not None else 0,
        a.stride(2) if seq_idx is not None else 0,
        out.stride(2) if seq_idx is not None else 0,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        IS_CAUSAL=causal, HAS_SEQ_IDX=seq_idx is not None
    )

    return out
