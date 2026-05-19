import torch
import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr, seq_idx_ptr,
    B, M, N, K,
    G, H,
    num_chunks,
    stride_a_batch, stride_a_group, stride_a_head, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_group, stride_b_head, stride_b_k, stride_b_n,
    stride_out_batch, stride_out_group, stride_out_head, stride_out_m, stride_out_n,
    IS_CAUSAL: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_h = tl.program_id(2)

    pid_batch = pid_bc // num_chunks
    pid_chunk = pid_bc % num_chunks

    off_m = pid_chunk * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_k = tl.arange(0, BLOCK_SIZE_K)

    a_offs = (
        pid_batch * stride_a_batch +
        pid_g * stride_a_group +
        pid_h * stride_a_head +
        off_m[:, None] * stride_a_m +
        off_k[None, :] * stride_a_k
    )
    b_offs = (
        pid_batch * stride_b_batch +
        pid_g * stride_b_group +
        pid_h * stride_b_head +
        off_k[:, None] * stride_b_k +
        off_n[None, :] * stride_b_n
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(
            a_ptr + a_offs,
            mask=(off_m[:, None] < M) & (k + off_k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptr + b_offs,
            mask=(k + off_k[:, None] < K) & (off_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_offs += BLOCK_SIZE_K * stride_a_k
        b_offs += BLOCK_SIZE_K * stride_b_k

    if IS_CAUSAL:
        row = pid_chunk * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
        col = tl.arange(0, BLOCK_SIZE_N)[None, :]
        mask = row >= col
        acc = tl.where(mask, acc, 0.0)

    if HAS_SEQ_IDX:
        seq_offs = pid_batch * M + off_m
        seq_m = tl.load(seq_idx_ptr + seq_offs, mask=off_m < M, other=-1)
        seq_n = tl.load(seq_idx_ptr + pid_batch * N + off_n, mask=off_n < N, other=-1)
        mask = seq_m[:, None] == seq_n[None, :]
        acc = tl.where(mask, acc, 0.0)

    out_offs = (
        pid_batch * stride_out_batch +
        pid_g * stride_out_group +
        pid_h * stride_out_head +
        off_m[:, None] * stride_out_m +
        off_n[None, :] * stride_out_n
    )
    tl.store(
        out_ptr + out_offs,
        acc,
        mask=(off_m[:, None] < M) & (off_n[None, :] < N)
    )

def _bmm_chunk_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    seq_idx: torch.Tensor = None,
    causal: bool = False,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
    BLOCK_SIZE_K: int = 32
):
    assert a.dim() == 5 and b.dim() == 5, "Inputs must be 5D tensors"
    B, G, H, M, K = a.shape
    B1, G1, H1, K1, N = b.shape
    assert B == B1 and G == G1 and H == H1 and K == K1, "Input shapes mismatch"

    out = torch.empty((B, G, H, M, N), device=a.device, dtype=a.dtype)
    num_chunks = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid = (B * num_chunks, G, H)

    seq_idx_ptr = a.data_ptr() if seq_idx is None else seq_idx.data_ptr()

    _bmm_chunk_fwd_kernel[grid](
        a, b, out, seq_idx_ptr,
        B, M, N, K, G, H, num_chunks,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3), a.stride(4),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3), b.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        IS_CAUSAL=causal,
        HAS_SEQ_IDX=seq_idx is not None,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return out
