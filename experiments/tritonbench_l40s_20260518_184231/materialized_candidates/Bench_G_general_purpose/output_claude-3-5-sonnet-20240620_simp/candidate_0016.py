import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, B_Start_Loc, B_Seqlen, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, WINDOW_SIZE: tl.constexpr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_on,
    batch_size, num_heads, seqlen_max, dim
):
    # Program ID
    pid = tl.program_id(0)
    batch_id = pid // num_heads
    head_id = pid % num_heads

    # Initialize offsets
    start_loc = tl.load(B_Start_Loc + batch_id)
    seqlen_q = tl.load(B_Seqlen + batch_id)

    # Block pointers
    q_offset = start_loc * stride_qm + head_id * stride_qh
    k_offset = start_loc * stride_kn + head_id * stride_kh
    v_offset = start_loc * stride_vn + head_id * stride_vh
    o_offset = start_loc * stride_om + head_id * stride_oh

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    max_score = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    sum_score = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load Q block
    q_ptrs = q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qm + \
             tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk
    q = tl.load(Q + q_ptrs, mask=tl.arange(0, BLOCK_M)[:, None] < seqlen_q, other=0.0)

    # Iterate over K,V blocks
    for block_n in range(0, seqlen_q, BLOCK_N):
        # Load K block
        k_ptrs = k_offset + tl.arange(0, BLOCK_N)[:, None] * stride_kn + \
                 tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kk
        k = tl.load(K + k_ptrs, 
                   mask=tl.arange(block_n, block_n + BLOCK_N)[:, None] < seqlen_q,
                   other=0.0)

        # Compute attention scores
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale

        # Apply sliding window attention mask
        window_mask = tl.abs(tl.arange(0, BLOCK_M)[:, None] - \
                    (block_n + tl.arange(0, BLOCK_N)[None, :])) <= WINDOW_SIZE
        qk = tl.where(window_mask, qk, float("-inf"))

        # Update max scores and compute exponentials
        block_max = tl.max(qk, 1)
        max_score_new = tl.maximum(max_score, block_max)
        exp_qk = tl.exp(qk - max_score_new[:, None])
        exp_qk = tl.where(window_mask, exp_qk, 0.0)

        # Load V block and compute weighted sum
        v_ptrs = v_offset + tl.arange(0, BLOCK_N)[:, None] * stride_vn + \
                 tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vk
        v = tl.load(V + v_ptrs,
                   mask=tl.arange(block_n, block_n + BLOCK_N)[:, None] < seqlen_q,
                   other=0.0)

        # Update accumulators
        sum_score_new = sum_score * tl.exp(max_score - max_score_new) + \
                       tl.sum(exp_qk, 1)
        acc_scale = tl.exp(max_score - max_score_new) / sum_score_new[:, None]
        acc = acc * acc_scale[:, None]
        acc += tl.dot(exp_qk, v) / sum_score_new[:, None]

        # Update running statistics
        max_score = max_score_new
        sum_score = sum_score_new

    # Write output
    o_ptrs = o_offset + tl.arange(0, BLOCK_M)[:, None] * stride_om + \
             tl.arange(0, BLOCK_DMODEL)[None, :] * stride_on
    tl.store(Out + o_ptrs, acc, mask=tl.arange(0, BLOCK_M)[:, None] < seqlen_q)

def context_attention_fwd(q, k, v, b_start_loc, b_seqlen, window_size=None):
    batch_size, num_heads, seqlen_max, dim = q.shape
    
    # Set default window size if not provided
    if window_size is None:
        window_size = seqlen_max
    
    # Compute scale factor
    sm_scale = 1.0 / (dim ** 0.5)
    
    # Allocate output tensor
    out = torch.empty_like(q)
    
    # Configure kernel parameters
    BLOCK = 128
    num_warps = 4
    
    # Launch kernel
    grid = (batch_size * num_heads,)
    _fwd_kernel[grid](
        q, k, v, out, b_start_loc, b_seqlen, sm_scale,
        BLOCK, BLOCK, dim, window_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, num_heads, seqlen_max, dim,
        num_warps=num_warps
    )
    
    return out
