import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, a_batch_stride, a_chunk_stride, a_head_stride, a_m_stride, a_k_stride,
    b_ptr, b_batch_stride, b_chunk_stride, b_head_stride, b_k_stride, b_n_stride,
    out_ptr, out_batch_stride, out_chunk_stride, out_head_stride, out_m_stride, out_n_stride,
    seq_a_ptr, seq_b_ptr, seq_a_stride, seq_b_stride,
    M, N, K, B, num_heads, num_chunks,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr
):
    # -------------------------------------------------------------
    # pids for batch, chunk, head, etc.
    # -------------------------------------------------------------
    pid = tl.program_id(0)
    pid_b = pid // (num_heads * num_chunks)
    rr = pid % (num_heads * num_chunks)
    pid_h = rr // num_chunks
    pid_c = rr % num_chunks

    # program_id(1) and program_id(2) for M and N blocks
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute block starting indices
    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N

    # Exits if outside the boundary
    if (offs_m >= M) or (offs_n >= N) or (pid_b >= B) or (pid_h >= num_heads) or (pid_c >= num_chunks):
        return

    # -------------------------------------------------------------
    # Base pointers
    # -------------------------------------------------------------
    a_block_ptr = a_ptr + pid_b * a_batch_stride + pid_c * a_chunk_stride + pid_h * a_head_stride
    b_block_ptr = b_ptr + pid_b * b_batch_stride + pid_c * b_chunk_stride + pid_h * b_head_stride
    out_block_ptr = out_ptr + pid_b * out_batch_stride + pid_c * out_chunk_stride + pid_h * out_head_stride

    # Offsets for M, N
    a_offset = a_block_ptr + offs_m * a_m_stride
    out_offset = out_block_ptr + offs_m * out_m_stride + offs_n * out_n_stride

    # -------------------------------------------------------------
    # Initialize ACC
    # -------------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -------------------------------------------------------------
    # BMM accumulation
    # -------------------------------------------------------------
    # Each iteration loads a tile from A and B, then computes partial matmul
    for k_offs in range(0, K, BLOCK_SIZE_K):
        # Load sub-block from A
        a_offs_k = a_offset + k_offs * a_k_stride
        a_tile = tl.load(
            a_offs_k + (tl.arange(0, BLOCK_SIZE_M)[:, None] * a_m_stride)
                      + (tl.arange(0, BLOCK_SIZE_K)[None, :] * a_k_stride),
            mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] + offs_m < M) &
                 (tl.arange(0, BLOCK_SIZE_K)[None, :] + k_offs < K),
            other=0.0
        )

        # Load sub-block from B
        b_offs_k = b_block_ptr + k_offs * b_k_stride + offs_n * b_n_stride
        b_tile = tl.load(
            b_offs_k + (tl.arange(0, BLOCK_SIZE_K)[:, None] * b_k_stride)
                      + (tl.arange(0, BLOCK_SIZE_N)[None, :] * b_n_stride),
            mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] + k_offs < K) &
                 (tl.arange(0, BLOCK_SIZE_N)[None, :] + offs_n < N),
            other=0.0
        )

        # Multiply
        partial = tl.dot(a_tile, b_tile)
        acc += partial

    # -------------------------------------------------------------
    # Causal masking (if IS_CAUSAL)
    # -------------------------------------------------------------
    if IS_CAUSAL:
        row_idx = tl.arange(0, BLOCK_SIZE_M)[:, None] + offs_m
        col_idx = tl.arange(0, BLOCK_SIZE_N)[None, :] + offs_n
        mask = col_idx >= row_idx
        acc = tl.where(mask, acc, 0.0)

    # -------------------------------------------------------------
    # Sequence index masking (if HAS_SEQ_IDX)
    # -------------------------------------------------------------
    if HAS_SEQ_IDX:
        # Load sequence IDs
        seq_a_val = tl.load(
            seq_a_ptr + pid_b * seq_a_stride + (offs_m + tl.arange(0, BLOCK_SIZE_M))
        )
        seq_b_val = tl.load(
            seq_b_ptr + pid_b * seq_b_stride + (offs_n + tl.arange(0, BLOCK_SIZE_N))
        )
        seq_a_val_broadcast = seq_a_val[:, None]
        seq_b_val_broadcast = seq_b_val[None, :]

        seq_mask = seq_a_val_broadcast == seq_b_val_broadcast
        acc = tl.where(seq_mask, acc, 0.0)

    # -------------------------------------------------------------
    # Store the result
    # -------------------------------------------------------------
    out_mask_m = tl.arange(0, BLOCK_SIZE_M)[:, None] + offs_m < M
    out_mask_n = tl.arange(0, BLOCK_SIZE_N)[None, :] + offs_n < N
    out_mask = out_mask_m & out_mask_n
    tl.store(
        out_offset + (tl.arange(0, BLOCK_SIZE_M)[:, None] * out_m_stride)
                   + (tl.arange(0, BLOCK_SIZE_N)[None, :] * out_n_stride),
        acc,
        mask=out_mask
    )

def _bmm_chunk_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor],
    seq_a: Optional[torch.Tensor],
    seq_b: Optional[torch.Tensor],
    *,
    M: int,
    N: int,
    K: int,
    B: int,
    num_heads: int,
    num_chunks: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    is_causal: bool = False,
    has_seq_idx: bool = False
):
    # Ensure output is allocated if not provided
    if out is None:
        out = torch.empty((B, num_chunks, num_heads, M, N), dtype=a.dtype, device=a.device)

    # Get pointer and strides
    a_ptr = a.data_ptr()
    b_ptr = b.data_ptr()
    out_ptr = out.data_ptr()

    a_strides = a.stride()
    b_strides = b.stride()
    out_strides = out.stride()

    # For sequence indices
    seq_a_ptr = seq_a.data_ptr() if seq_a is not None else 0
    seq_b_ptr = seq_b.data_ptr() if seq_b is not None else 0

    # Prepare kernel grid
    grid = (B * num_heads * num_chunks, (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)

    # Launch kernel
    _bmm_chunk_fwd_kernel[grid](
        a_ptr,
        a_strides[0],  # batch stride
        a_strides[1],  # chunk stride
        a_strides[2],  # head stride
        a_strides[3],  # m stride
        a_strides[4],  # k stride
        b_ptr,
        b_strides[0],
        b_strides[1],
        b_strides[2],
        b_strides[3],
        b_strides[4],
        out_ptr,
        out_strides[0],
        out_strides[1],
        out_strides[2],
        out_strides[3],
        out_strides[4],
        seq_a_ptr,
        seq_b_ptr,
        seq_a.stride(0) if seq_a is not None else 0,
        seq_b.stride(0) if seq_b is not None else 0,
        M,
        N,
        K,
        B,
        num_heads,
        num_chunks,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        IS_CAUSAL=is_causal,
        HAS_SEQ_IDX=has_seq_idx
    )

    return out
