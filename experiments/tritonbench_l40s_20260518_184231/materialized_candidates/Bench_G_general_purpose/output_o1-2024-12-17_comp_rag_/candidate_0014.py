import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    B_Start_Loc, B_Seqlen,
    sm_scale,
    stride_q_bs, stride_q_h, stride_q_m,
    stride_k_bs, stride_k_h, stride_k_m,
    stride_v_bs, stride_v_h, stride_v_m,
    stride_o_bs, stride_o_h, stride_o_m,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    # Each program processes BLOCK_M tokens in the "M" dimension
    m_block_idx = tl.program_id(2)

    # Offsets for batch start and seqlen
    batch_start = tl.load(B_Start_Loc + bid)
    seqlen = tl.load(B_Seqlen + bid)

    # Current token range this warp operates on
    m_offset = m_block_idx * BLOCK_M
    block_m_range = tl.arange(0, BLOCK_M)
    m_indices = m_offset + block_m_range

    # Pointers to Q, K, V, Out
    q_ptrs = Q + bid * stride_q_bs + hid * stride_q_h + m_indices[:, None] * stride_q_m
    out_ptrs = Out + bid * stride_o_bs + hid * stride_o_h + m_indices[:, None] * stride_o_m

    # Initialize partial sums
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    m_valid = m_indices < seqlen

    # Sliding window loop
    # For demonstration, we assume a fixed window size "window" blocks around the current M block
    # Adjust as needed for real usage
    window_blocks = 4
    for n_block_off in range(-window_blocks, window_blocks + 1):
        n_block_idx = m_block_idx + n_block_off
        if n_block_idx < 0 or n_block_idx * BLOCK_N >= seqlen:
            continue
        n_offset = n_block_idx * BLOCK_N
        n_range = tl.arange(0, BLOCK_N)
        n_indices = n_offset + n_range
        n_valid = n_indices < seqlen

        # Load K, V
        k_ptrs = K + bid * stride_k_bs + hid * stride_k_h + n_indices[None, :] * stride_k_m
        v_ptrs = V + bid * stride_v_bs + hid * stride_v_h + n_indices[None, :] * stride_v_m

        # Gather Q, K
        q = tl.load(q_ptrs + tl.arange(0, BLOCK_M)[:, None] * 1, mask=m_valid[:, None], other=0.0)
        k = tl.load(k_ptrs + tl.arange(0, BLOCK_DMODEL)[:, None] * 1, mask=n_valid[None, :], other=0.0)
        k = tl.transpose(k, [1, 0])

        # Compute QK^T
        scores = tl.dot(q, k) * sm_scale

        # Apply mask for invalid tokens
        scores_mask = m_valid[:, None] & n_valid[None, :]
        scores = tl.where(scores_mask, scores, float('-inf'))

        # Softmax
        scores_max = tl.max(scores, 1)
        scores_exp = tl.exp(scores - scores_max[:, None])
        denom = tl.sum(scores_exp, 1)
        attn = scores_exp / denom[:, None]

        # Gather V
        v = tl.load(v_ptrs + tl.arange(0, BLOCK_DMODEL)[:, None] * 1, mask=n_valid[None, :], other=0.0)
        v = tl.transpose(v, [1, 0])

        # Accumulate partial output
        acc += tl.dot(attn, v)

    # Write to Out
    tl.store(out_ptrs + tl.arange(0, BLOCK_M)[:, None] * 1, acc, mask=m_valid[:, None])


def context_attention_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    B_Start_Loc: torch.Tensor,
    B_Seqlen: torch.Tensor,
    sm_scale: float,
    BLOCK_M=64,
    BLOCK_DMODEL=64,
    BLOCK_N=64
):
    batch = Q.shape[0]
    heads = Q.shape[1]
    seqlen = Q.shape[2]
    d_model = Q.shape[3]

    Out = torch.empty_like(Q)

    grid = (batch, heads, (seqlen + BLOCK_M - 1) // BLOCK_M)
    triton.run(
        _fwd_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        args=[
            Q, K, V, Out,
            B_Start_Loc, B_Seqlen,
            sm_scale,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2)
        ],
        constants=[BLOCK_M, BLOCK_DMODEL, BLOCK_N]
    )
    return Out
