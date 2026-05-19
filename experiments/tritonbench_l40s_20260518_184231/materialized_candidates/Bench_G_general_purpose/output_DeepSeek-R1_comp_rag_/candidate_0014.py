import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    B_Start_Loc, B_Seqlen,
    Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    # Program IDs for batch, head, and query block
    b = tl.program_id(0)
    h = tl.program_id(1)
    block_m = tl.program_id(2)

    # Load batch metadata
    batch_start = tl.load(B_Start_Loc + b)
    seq_len = tl.load(B_Seqlen + b)
    # Current query block boundaries
    q_start = batch_start + block_m * BLOCK_M
    q_end = tl.minimum(q_start + BLOCK_M, batch_start + seq_len)

    # Initialize accumulators for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load query block
    offs_m = tl.arange(0, BLOCK_M)
    offs_q = (q_start + offs_m)[:, None] * stride_qm + h * stride_qh + b * stride_qb
    q = tl.load(Q + offs_q, mask=(q_start + offs_m < q_end)[:, None], other=0.0)

    # Determine key window boundaries
    window_start = tl.maximum(q_start - WINDOW_SIZE, batch_start)
    window_end = q_end
    num_n_blocks = tl.cdiv(window_end - window_start, BLOCK_N)

    # Process key blocks within window
    for block_n in range(num_n_blocks):
        n_start = window_start + block_n * BLOCK_N
        n_end = tl.minimum(n_start + BLOCK_N, window_end)

        # Load K and V blocks
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = (n_start + offs_n)[None, :] * stride_kn + h * stride_kh + b * stride_kb
        k = tl.load(K + offs_k, mask=(n_start + offs_n < n_end)[None, :], other=0.0)
        
        offs_v = (n_start + offs_n)[:, None] * stride_vn + h * stride_vh + b * stride_vb
        v = tl.load(V + offs_v, mask=(n_start + offs_n < n_end)[:, None], other=0.0)

        # Compute attention scores
        scores = tl.dot(q, tl.trans(k.to(q.dtype))) * sm_scale

        # Generate sliding window mask
        q_pos = q_start + offs_m
        k_pos = n_start + offs_n
        mask = (k_pos[None, :] <= q_pos[:, None]) & (k_pos[None, :] >= (q_pos[:, None] - WINDOW_SIZE))
        scores = tl.where(mask, scores, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i[:, None], tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        exp_scores = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(exp_scores, axis=1)
        acc = acc * alpha[:, None] + tl.dot(exp_scores.to(v.dtype), v)
        m_i = m_new

    # Normalize and store output
    acc = acc / l_i[:, None]
    offs_om = (q_start + offs_m)[:, None] * stride_om + h * stride_oh + b * stride_ob
    tl.store(Out + offs_om, acc.to(Out.dtype.element_ty), mask=(q_start + offs_m < q_end)[:, None])

def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    B_Start_Loc: torch.Tensor,
    B_Seqlen: torch.Tensor,
    sm_scale: float,
    WINDOW_SIZE: int = 512,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
):
    assert q.shape[-1] == k.shape[-1], "Q/K feature dimensions must match"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    
    # Kernel configuration
    device = q.device
    grid = (len(B_Start_Loc), q.size(1), triton.cdiv(q.size(2), BLOCK_M))
    
    _fwd_kernel[grid](
        q, k, v, sm_scale,
        B_Start_Loc, B_Seqlen,
        o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=q.size(-1),
        BLOCK_N=BLOCK_N,
        WINDOW_SIZE=WINDOW_SIZE,
    )
    return o
