import torch
import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_fwd_kernel(
    a, b, out,
    seq_idx_ptr,
    batch_size,
    chunk_size,
    group_size,
    head_size,
    dim,
    stride_batch_a, stride_chunk_a, stride_group_a, stride_head_a, stride_dim_a,
    stride_batch_b, stride_chunk_b, stride_group_b, stride_head_b, stride_dim_b,
    stride_batch_out, stride_chunk_out, stride_group_out, stride_head_out, stride_dim_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr
):
    pid = tl.program_id(0)
    chunk_id = pid % chunk_size
    group_id = (pid // chunk_size) % group_size
    head_id = (pid // chunk_size // group_size) % head_size
    batch_id = pid // chunk_size // group_size // head_size
    offs_m = (batch_id * stride_batch_a +
              chunk_id * stride_chunk_a +
              group_id * stride_group_a +
              head_id * stride_head_a +
              tl.arange(0, BLOCK_SIZE_M))
    offs_n = (batch_id * stride_batch_b +
              chunk_id * stride_chunk_b +
              group_id * stride_group_b +
              head_id * stride_head_b +
              tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a + offs_m[:, None] * stride_dim_a + offs_k[None, :]
    b_ptrs = b + offs_k[:, None] * stride_dim_b + offs_n[None, :]
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, dim, BLOCK_SIZE_K):
        k_off = k // BLOCK_SIZE_K
        a_val = tl.load(a_ptrs, mask=(offs_m[:, None] < batch_size * stride_batch_a) &
                        (offs_k[None, :] < (dim - k) * stride_dim_a), other=0.0)
        b_val = tl.load(b_ptrs, mask=(offs_k[:, None] < (dim - k) * stride_dim_b) &
                        (offs_n[None, :] < dim * stride_dim_b), other=0.0)
        accumulator += tl.dot(a_val, b_val)
        a_ptrs += BLOCK_SIZE_K * stride_dim_a
        b_ptrs += BLOCK_SIZE_K * stride_dim_b
    if IS_CAUSAL:
        causal_mask = tl.arange(0, BLOCK_SIZE_M)[:, None] >= tl.arange(0, BLOCK_SIZE_N)[None, :]
        accumulator = tl.where(causal_mask, accumulator, 0)
    out_offs = (batch_id * stride_batch_out +
                chunk_id * stride_chunk_out +
                group_id * stride_group_out +
                head_id * stride_head_out +
                tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_dim_out +
                tl.arange(0, BLOCK_SIZE_N)[None, :])
    out_mask = (batch_id * stride_batch_out +
                chunk_id * stride_chunk_out +
                group_id * stride_group_out +
                head_id * stride_head_out +
                tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size * stride_batch_out) & \
               (batch_id * stride_batch_out +
                chunk_id * stride_chunk_out +
                group_id * stride_group_out +
                head_id * stride_head_out +
                tl.arange(0, BLOCK_SIZE_N)[None, :] < batch_size * stride_batch_out)
    tl.store(out + out_offs, accumulator, mask=out_mask)
    if HAS_SEQ_IDX:
        seq_idx = tl.load(seq_idx_ptr + batch_id)
        out_offs_seq_idx_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < seq_idx) & \
                                (tl.arange(0, BLOCK_SIZE_N)[None, :] < seq_idx)
        tl.store(out + out_offs, 0, mask=out_offs_seq_idx_mask)


def _bmm_chunk_fwd(a, b, *,
                   out=None,
                   chunk_size=1,
                   group_size=1,
                   head_size=1,
                   dim=None,
                   seq_idx=None,
                   is_causal=False):
    if a.stride(0) != 1 and a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(0) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    batch_size, dim = a.shape
    if dim != b.shape[1]:
        raise ValueError(f"Incompatible dimensions, got {a.shape} and {b.shape}")
    if out is None:
        out = torch.empty((batch_size, chunk_size, group_size, head_size, dim),
                          dtype=a.dtype, device=a.device)
    else:
        out = out.reshape((batch_size, chunk_size, group_size, head_size, dim))
        if out.shape != (batch_size, chunk_size, group_size, head_size, dim):
            raise ValueError(f"Incompatible out tensors, expected {(batch_size, chunk_size, group_size, head_size, dim)} "
                             f"but got {out.shape}")
    if seq_idx is not None:
        seq_idx_arg = seq_idx.to(torch.int32)
    else:
        seq_idx_arg = torch.tensor(0, dtype=torch.int32, device=a.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_M) *
            triton.cdiv(chunk_size, BLOCK_SIZE_N) *
            group_size * head_size)
    _bmm_chunk_fwd_kernel[grid](
        a, b, out,
        seq_idx_arg,
        batch_size, chunk_size, group_size, head_size, dim,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3), a.stride(4),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3), b.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
