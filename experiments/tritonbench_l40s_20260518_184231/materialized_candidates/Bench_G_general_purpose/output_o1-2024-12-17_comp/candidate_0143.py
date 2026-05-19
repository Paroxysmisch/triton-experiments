import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr, 
    seqA_ptr, seqB_ptr,
    BATCH, HEAD, CHUNK,
    M, N, K,
    stride_a_b, stride_a_h, stride_a_c, stride_a_m, stride_a_k,
    stride_b_b, stride_b_h, stride_b_c, stride_b_k, stride_b_n,
    stride_o_b, stride_o_h, stride_o_c, stride_o_m, stride_o_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr
):
    pid = tl.program_id(0)
    batch_head_chunk = tl.program_id(1)

    # Decompose program_id(1) into batch, head, chunk indices
    batch_idx = batch_head_chunk // (HEAD * CHUNK)
    rem = batch_head_chunk % (HEAD * CHUNK)
    head_idx = rem // CHUNK
    chunk_idx = rem % CHUNK

    # Decompose pid into block_m and block_n
    block_m = pid // ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    block_n = pid % ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)

    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Pointers to seqA, seqB if needed
    # For simplicity, assume seq lengths match M, N dims respectively
    seqA_offs = offs_m + chunk_idx * M if HAS_SEQ_IDX else 0
    seqB_offs = offs_n + chunk_idx * N if HAS_SEQ_IDX else 0

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in steps of BLOCK_SIZE_K
    for k_block in range(0, K, BLOCK_SIZE_K):
        offs_k = k_block + tl.arange(0, BLOCK_SIZE_K)

        # Compute memory locations for the sub-blocks
        a_ptrs = a_ptr + (
            batch_idx * stride_a_b
            + head_idx * stride_a_h
            + chunk_idx * stride_a_c
            + offs_m[:, None] * stride_a_m
            + offs_k[None, :] * stride_a_k
        )
        b_ptrs = b_ptr + (
            batch_idx * stride_b_b
            + head_idx * stride_b_h
            + chunk_idx * stride_b_c
            + offs_k[:, None] * stride_b_k
            + offs_n[None, :] * stride_b_n
        )

        # Load blocks
        a_block = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b_block = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Accumulate
        acc += tl.dot(a_block, b_block)

    # Apply causal mask if needed
    if IS_CAUSAL:
        # Positions in M and N
        m_pos = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
        n_pos = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
        causal_mask = n_pos >= m_pos
        acc = tl.where(causal_mask, acc, 0.0)

    # Apply sequence index mask if needed
    if HAS_SEQ_IDX:
        seqA_vals = tl.load(seqA_ptr + seqA_offs, mask=(seqA_offs < M), other=0)[:, None]
        seqB_vals = tl.load(seqB_ptr + seqB_offs, mask=(seqB_offs < N), other=0)[None, :]
        valid_mask = (seqA_vals == seqB_vals) & (seqA_offs[:, None] < M) & (seqB_offs[None, :] < N)
        acc = tl.where(valid_mask, acc, 0.0)

    # Store
    out_ptrs = out_ptr + (
        batch_idx * stride_o_b
        + head_idx * stride_o_h
        + chunk_idx * stride_o_c
        + offs_m[:, None] * stride_o_m
        + offs_n[None, :] * stride_o_n
    )
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def _bmm_chunk_fwd(
    a, b, 
    out,
    seqA, seqB,
    BATCH, HEAD, CHUNK,
    M, N, K,
    stride_a_b, stride_a_h, stride_a_c, stride_a_m, stride_a_k,
    stride_b_b, stride_b_h, stride_b_c, stride_b_k, stride_b_n,
    stride_o_b, stride_o_h, stride_o_c, stride_o_m, stride_o_n,
    BLOCK_SIZE_M=64,
    BLOCK_SIZE_N=64,
    BLOCK_SIZE_K=32,
    IS_CAUSAL=False,
    HAS_SEQ_IDX=False
):
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m * grid_n, BATCH * HEAD * CHUNK)
    _bmm_chunk_fwd_kernel[grid](
        a, b, out,
        seqA, seqB,
        BATCH, HEAD, CHUNK,
        M, N, K,
        stride_a_b, stride_a_h, stride_a_c, stride_a_m, stride_a_k,
        stride_b_b, stride_b_h, stride_b_c, stride_b_k, stride_b_n,
        stride_o_b, stride_o_h, stride_o_c, stride_o_m, stride_o_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        IS_CAUSAL, HAS_SEQ_IDX
    )
    return out
