import triton
import triton.language as tl
import torch

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr, seq_idx_ptr,
    batch_size, num_chunks, num_groups, num_heads,
    M, N, K,
    stride_a_batch, stride_a_chunk, stride_a_group, stride_a_head, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_chunk, stride_b_group, stride_b_head, stride_b_k, stride_b_n,
    stride_out_batch, stride_out_chunk, stride_out_group, stride_out_head, stride_out_m, stride_out_n,
    stride_seq_batch, stride_seq_chunk, stride_seq_m,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr, HAS_SEQ_IDX: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    pid_batch = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_combined = tl.program_id(2)
    
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    gid = pid_combined // (num_heads * num_blocks_m * num_blocks_n)
    pid_head = (pid_combined // (num_blocks_m * num_blocks_n)) % num_heads
    pid_group = gid % num_groups
    
    block_m = (pid_combined // num_blocks_n) % num_blocks_m
    block_n = pid_combined % num_blocks_n
    
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_offs = (pid_batch * stride_a_batch + pid_chunk * stride_a_chunk +
              pid_group * stride_a_group + pid_head * stride_a_head +
              offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k)
    
    b_offs = (pid_batch * stride_b_batch + pid_chunk * stride_b_chunk +
              pid_group * stride_b_group + pid_head * stride_b_head +
              offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptr + a_offs, mask=(offs_k[None, :] < K - k) & (offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptr + b_offs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_offs += BLOCK_SIZE_K * stride_a_k
        b_offs += BLOCK_SIZE_K * stride_b_k
    
    if IS_CAUSAL:
        causal_mask = (offs_m[:, None] >= offs_n[None, :])
        acc = tl.where(causal_mask, acc, 0.0)
    
    if HAS_SEQ_IDX:
        seq_offs_a = (pid_batch * stride_seq_batch + pid_chunk * stride_seq_chunk + offs_m)
        seq_idx_a = tl.load(seq_idx_ptr + seq_offs_a, mask=offs_m < M, other=-1)
        seq_offs_b = (pid_batch * stride_seq_batch + pid_chunk * stride_seq_chunk + offs_n)
        seq_idx_b = tl.load(seq_idx_ptr + seq_offs_b, mask=offs_n < N, other=-1)
        seq_mask = (seq_idx_a[:, None] == seq_idx_b[None, :])
        acc = tl.where(seq_mask, acc, 0.0)
    
    out_offs = (pid_batch * stride_out_batch + pid_chunk * stride_out_chunk +
                pid_group * stride_out_group + pid_head * stride_out_head +
                offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    tl.store(out_ptr + out_offs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def _bmm_chunk_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    seq_idx: Optional[torch.Tensor] = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert a.dim() == 6 and b.dim() == 6, "a and b must be 6D tensors"
    batch_size, num_chunks, num_groups, num_heads, M, K = a.shape
    _, _, _, _, K_, N = b.shape
    assert K == K_, "Incompatible K dimensions"
    
    if out is None:
        out = torch.empty((batch_size, num_chunks, num_groups, num_heads, M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 16, 16
    
    grid = (batch_size, num_chunks, num_groups * num_heads * (triton.cdiv(M, BLOCK_SIZE_M)) * (triton.cdiv(N, BLOCK_SIZE_N)))
    
    has_seq_idx = seq_idx is not None
    if has_seq_idx:
        assert seq_idx.shape == (batch_size, num_chunks, M), "seq_idx must have shape (batch, num_chunks, M)"
    
    _bmm_chunk_fwd_kernel[grid](
        a, b, out, seq_idx,
        batch_size, num_chunks, num_groups, num_heads, M, N, K,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3), a.stride(4), a.stride(5),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3), b.stride(4), b.stride(5),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4), out.stride(5),
        seq_idx.stride(0) if has_seq_idx else 0,
        seq_idx.stride(1) if has_seq_idx else 0,
        seq_idx.stride(2) if has_seq_idx else 0,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        IS_CAUSAL=causal,
        HAS_SEQ_IDX=has_seq_idx,
    )
    
    return out
