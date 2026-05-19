import triton
import triton.language as tl
import torch

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr, seq_idx_ptr,
    a_stride_m, a_stride_k, b_stride_k, b_stride_n, out_stride_m, out_stride_n, seq_idx_stride,
    batch_size, chunk_size, group_size, head_size, seq_len, seq_idx_len,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr, HAS_SEQ_IDX: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (chunk_size * group_size * head_size)
    chunk_id = (pid // (group_size * head_size)) % chunk_size
    group_id = (pid // head_size) % group_size
    head_id = pid % head_size

    m = batch_id * seq_len + chunk_id * BLOCK_SIZE_M
    n = group_id * BLOCK_SIZE_N
    k = head_id * BLOCK_SIZE_K

    a_base = a_ptr + batch_id * a_stride_m * seq_len + chunk_id * a_stride_m * BLOCK_SIZE_M
    b_base = b_ptr + batch_id * b_stride_n * seq_len + group_id * b_stride_n * BLOCK_SIZE_N
    out_base = out_ptr + batch_id * out_stride_m * seq_len + group_id * out_stride_n * BLOCK_SIZE_N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for b in range(0, BLOCK_SIZE_K, BLOCK_SIZE_K):
        a = tl.load(a_base + m * a_stride_m + (k + b) * a_stride_k, mask=(m < seq_len and k + b < seq_len), other=0.0)
        b = tl.load(b_base + (k + b) * b_stride_k + n * b_stride_n, mask=(k + b < seq_len and n < seq_len), other=0.0)
        acc += a[:, None] * b[None, :]
    
    if IS_CAUSAL:
        causal_mask = tl.arange(BLOCK_SIZE_M)[:, None] < tl.arange(BLOCK_SIZE_N)[None, :]
        acc = tl.where(causal_mask, acc, 0.0)

    if HAS_SEQ_IDX:
        seq_idx_base = seq_idx_ptr + batch_id * seq_idx_stride
        seq_idx = tl.load(seq_idx_base + seq_idx_len * group_id + head_id, mask=(group_id < group_size and head_id < head_size), other=0)
        valid_idx = tl.arange(BLOCK_SIZE_M) < seq_idx
        acc = tl.where(valid_idx, acc, 0.0)

    tl.store(out_base + m * out_stride_m + n * out_stride_n, acc, mask=(m < seq_len and n < seq_len))

def _bmm_chunk_fwd(
    a, b, seq_idx=None,
    BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
    IS_CAUSAL=False, HAS_SEQ_IDX=False
):
    # Determine the shapes and strides
    batch_size, seq_len, head_size, _, _ = a.shape
    group_size = 1  # Assuming group_size is always 1 for simplicity

    a_stride_m = a.stride(1)
    a_stride_k = a.stride(2)
    b_stride_k = b.stride(2)
    b_stride_n = b.stride(3)
    out_stride_m = b.stride(1)
    out_stride_n = b.stride(3)
    seq_idx_stride = seq_idx.stride(0) if seq_idx is not None else 0

    # Allocate output tensor
    out_shape = (batch_size, seq_len, group_size, head_size, BLOCK_SIZE_N)
    out = torch.empty(out_shape, device=a.device, dtype=a.dtype)

    # Determine grid size
    grid_size = (batch_size * chunk_size * group_size * head_size, 1, 1)

    # Launch the kernel
    _bmm_chunk_fwd_kernel[grid_size](
        a.data_ptr(), b.data_ptr(), out.data_ptr(), seq_idx.data_ptr() if seq_idx is not None else 0,
        a_stride_m, a_stride_k, b_stride_k, b_stride_n, out_stride_m, out_stride_n, seq_idx_stride,
        batch_size, chunk_size, group_size, head_size, seq_len, seq_idx.shape[0] if seq_idx is not None else 0,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        IS_CAUSAL, HAS_SEQ_IDX
    )

    return out
