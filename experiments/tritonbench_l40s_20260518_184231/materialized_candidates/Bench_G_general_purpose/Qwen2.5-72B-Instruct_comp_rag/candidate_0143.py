import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr,
    a_batch_stride, a_chunk_stride, a_group_stride, a_head_stride, a_m_stride, a_k_stride,
    b_batch_stride, b_chunk_stride, b_group_stride, b_head_stride, b_k_stride, b_n_stride,
    out_batch_stride, out_chunk_stride, out_group_stride, out_head_stride, out_m_stride, out_n_stride,
    M, N, K, chunk_size, group_size, head_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr, HAS_SEQ_IDX: tl.constexpr,
    seq_idx_ptr: tl.constexpr, seq_idx_stride: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    pid_mn = pid % num_pid_in_group
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_offs = (
        tl.program_id(axis=1) * a_chunk_stride +
        tl.program_id(axis=2) * a_group_stride +
        tl.program_id(axis=3) * a_head_stride +
        offs_m[:, None] * a_m_stride +
        offs_k[None, :] * a_k_stride
    )
    b_offs = (
        tl.program_id(axis=1) * b_chunk_stride +
        tl.program_id(axis=2) * b_group_stride +
        tl.program_id(axis=3) * b_head_stride +
        offs_k[:, None] * b_k_stride +
        offs_n[None, :] * b_n_stride
    )
    a_ptrs = a_ptr + a_offs
    b_ptrs = b_ptr + b_offs

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)
        acc += tl.dot(a_block, b_block)
        a_ptrs += BLOCK_SIZE_K * a_k_stride
        b_ptrs += BLOCK_SIZE_K * b_k_stride

    if IS_CAUSAL:
        mask = offs_m[:, None] >= offs_n[None, :]
        acc = tl.where(mask, acc, float('-inf'))

    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + tl.program_id(axis=1) * seq_idx_stride)
        mask = seq_idx == tl.program_id(axis=2)
        acc = tl.where(mask, acc, 0.0)

    out_offs = (
        tl.program_id(axis=1) * out_chunk_stride +
        tl.program_id(axis=2) * out_group_stride +
        tl.program_id(axis=3) * out_head_stride +
        offs_m[:, None] * out_m_stride +
        offs_n[None, :] * out_n_stride
    )
    out_ptrs = out_ptr + out_offs
    tl.store(out_ptrs, acc)

import torch
import triton
import triton.language as tl

def _bmm_chunk_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    chunk_size: int = 1,
    group_size: int = 1,
    head_size: int = 1,
    block_size_m: int = 16,
    block_size_n: int = 16,
    block_size_k: int = 16,
    is_causal: bool = False,
    has_seq_idx: bool = False,
    seq_idx: Optional[torch.Tensor] = None
):
    if out is None:
        out = torch.empty(
            (a.size(0), a.size(1), a.size(2), a.size(3), b.size(4)),
            dtype=a.dtype,
            device=a.device
        )

    grid = (
        (a.size(4) // block_size_m) * (b.size(5) // block_size_n),
        a.size(1),
        a.size(2),
        a.size(3)
    )

    seq_idx_ptr = seq_idx.data_ptr() if has_seq_idx else 0
    seq_idx_stride = seq_idx.stride(1) if has_seq_idx else 0

    _bmm_chunk_fwd_kernel[grid](
        a.data_ptr(), b.data_ptr(), out.data_ptr(),
        a.stride(0), a.stride(1), a.stride(2), a.stride(3), a.stride(4), a.stride(5),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3), b.stride(4), b.stride(5),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4), out.stride(5),
        a.size(4), b.size(5), a.size(5), chunk_size, group_size, head_size,
        BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n, BLOCK_SIZE_K=block_size_k,
        IS_CAUSAL=is_causal, HAS_SEQ_IDX=has_seq_idx,
        seq_idx_ptr=seq_idx_ptr, seq_idx_stride=seq_idx_stride
    )

    return out
