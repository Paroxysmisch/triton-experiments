import math
import triton
import triton.language as tl

# Define block sizes for partitioning the computation
BLOCK_M = 128        # Block size along the query length M dimension
BLOCK_N = 128        # Block size along the key length N dimension
BLOCK_DMODEL = 64    # Block size along the hidden dimension DMODEL

@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B_Start_Loc_ptr, B_Seqlen_ptr,
    sm_scale,  # precomputed scaling factor
    stride_q_batch, stride_q_head, stride_q_m, stride_q_d,
    stride_k_batch, stride_k_head, stride_k_n, stride_k_d,
    stride_v_batch, stride_v_head, stride_v_n, stride_v_d,
    stride_o_batch, stride_o_head, stride_o_m, stride_o_d,
    B, H, M, N, D,
    # Metaparameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Program IDs determine which portion of the batch/head (pid_bh) and query blocks (pid_m) we compute
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Convert (pid_bh) into batch_id, head_id by integer division/mod
    # Suppose each batch has H heads: pid_bh = batch_id * H + head_id
    batch_id = pid_bh // H
    head_id = pid_bh % H

    # Offsets in the query dimension for this block of M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Offsets in the depth dimension
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # We'll clamp these within actual size M for the partial block
    offs_m_clamped = tl.where(offs_m < M, offs_m, M - 1)

    # Fetch the start position and sequence length for this batch
    b_start = tl.load(B_Start_Loc_ptr + batch_id)
    b_seqlen = tl.load(B_Seqlen_ptr + batch_id)

    # Pointers to the base of Q, K, V for the current batch & head
    Q_base = Q_ptr + batch_id * stride_q_batch + head_id * stride_q_head
    K_base = K_ptr + batch_id * stride_k_batch + head_id * stride_k_head
    V_base = V_ptr + batch_id * stride_v_batch + head_id * stride_v_head
    Out_base = Out_ptr + batch_id * stride_o_batch + head_id * stride_o_head

    # Load Q for the entire BLOCK_M x BLOCK_DMODEL region
    q_ptrs = Q_base + offs_m_clamped[:, None] * stride_q_m + offs_d[None, :] * stride_q_d
    Q_block = tl.load(q_ptrs, mask=offs_m < M, other=0.0)

    # We'll compute the attention for each chunk of N dimension in a loop
    # Maintaining a partial maximum for stable softmax and partial sum of exps
    max_scores = tl.full([BLOCK_M], float("-inf"), tl.float32)
    exp_accum = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Each iteration covers a chunk of BLOCK_N along the key dimension
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        # Load K block
        k_ptrs = K_base + offs_n[None, :] * stride_k_n + offs_d[:, None] * stride_k_d
        K_block = tl.load(k_ptrs, mask=mask_n, other=0.0)
        # Dot product [BLOCK_M, BLOCK_DMODEL] x [BLOCK_DMODEL, BLOCK_N] => [BLOCK_M, BLOCK_N]
        # Q shape: (BLOCK_M, BLOCK_DMODEL), K shape: (BLOCK_DMODEL, BLOCK_N)
        scores = tl.dot(Q_block, K_block)
        # Scale
        scores = scores * sm_scale

        # We only consider valid tokens within the sequence length for each batch
        # Mask out positions beyond B_Seqlen or outside the chunk range
        valid_n_mask = (offs_n >= b_start) & (offs_n < b_start + b_seqlen)
        scores_mask = valid_n_mask[None, :]

        # Replace invalid positions with very negative for softmax
        scores = tl.where(scores_mask, scores, float("-inf"))

        # Update running max for stable softmax
        curr_max = tl.max(scores, 1)
        new_max = tl.maximum(max_scores, curr_max)
        # Compute exponent diff factor
        exp_diff_max = tl.exp(max_scores - new_max)
        exp_diff_curr = tl.exp(scores - new_max[:, None])

        # Update running sum of exps
        exp_accum = exp_accum * exp_diff_max + tl.sum(exp_diff_curr, 1)
        max_scores = new_max

    # Now we have the max_scores and sum of exps across all chunks of K
    # We'll do another loop to accumulate the final weighted sum by V
    inv_exp_accum = 1.0 / exp_accum

    # Accumulator for the output (BLOCK_M, BLOCK_DMODEL)
    out_block = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        # Load K (again) and V
        k_ptrs = K_base + offs_n[None, :] * stride_k_n + offs_d[:, None] * stride_k_d
        v_ptrs = V_base + offs_n[None, :] * stride_v_n + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_v_d

        K_block = tl.load(k_ptrs, mask=mask_n, other=0.0)
        V_block = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Dot product for the second pass
        scores = tl.dot(Q_block, K_block)
        scores *= sm_scale

        # Apply invalid positions mask
        valid_n_mask = (offs_n >= b_start) & (offs_n < b_start + b_seqlen)
        scores_mask = valid_n_mask[None, :]
        scores = tl.where(scores_mask, scores, float("-inf"))

        # Recompute stable exponent
        scores = tl.exp(scores - max_scores[:, None])
        # Weighted sum
        weights = scores * inv_exp_accum[:, None]
        out_block += tl.dot(weights, V_block)

    # Write out the final block of results
    out_ptrs = Out_base + offs_m_clamped[:, None] * stride_o_m + offs_d[None, :] * stride_o_d
    tl.store(out_ptrs, out_block, mask=offs_m < M)

def context_attention_fwd(Q, K, V, Out,
                          B_Start_Loc, B_Seqlen,
                          batch_size, heads, M, N, D):
    """
    Q, K, V, Out are contiguous memory buffers (e.g., torch.Tensor.data_ptr())
    B_Start_Loc, B_Seqlen are arrays containing start index and seq length for each batch
    """
    # Compute the softmax scale (inverse sqrt of D)
    sm_scale = 1.0 / math.sqrt(D)

    # Calculate the strides for Q, K, V, and Out
    # Assuming Q.shape = [batch_size, heads, M, D] => strides: 
    stride_q_batch = heads * M * D
    stride_q_head = M * D
    stride_q_m = D
    stride_q_d = 1

    # K.shape = [batch_size, heads, N, D]
    stride_k_batch = heads * N * D
    stride_k_head = N * D
    stride_k_n = D
    stride_k_d = 1

    # V.shape = [batch_size, heads, N, D]
    stride_v_batch = heads * N * D
    stride_v_head = N * D
    stride_v_n = D
    stride_v_d = 1

    # Out.shape = [batch_size, heads, M, D]
    stride_o_batch = heads * M * D
    stride_o_head = M * D
    stride_o_m = D
    stride_o_d = 1

    # Launch grid: each block handles a [BLOCK_M x BLOCK_DMODEL] chunk for a single (batch, head)
    # grid_x = number of blocks needed to cover M
    grid_x = (M + BLOCK_M - 1) // BLOCK_M
    # grid_y = batch_size * heads
    grid_y = batch_size * heads

    # Heuristic for number of warps
    # E.g., more warps if N is large
    num_warps = 4 if N <= 64 else 8

    _fwd_kernel[grid_x, grid_y](
        Q, K, V, Out,
        B_Start_Loc, B_Seqlen,
        sm_scale,
        stride_q_batch, stride_q_head, stride_q_m, stride_q_d,
        stride_k_batch, stride_k_head, stride_k_n, stride_k_d,
        stride_v_batch, stride_v_head, stride_v_n, stride_v_d,
        stride_o_batch, stride_o_head, stride_o_m, stride_o_d,
        batch_size, heads, M, N, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps
    )
