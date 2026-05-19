import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    sm_scale: tl.constexpr,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_m, stride_k_d,
    stride_v_b, stride_v_h, stride_v_m, stride_v_d,
    stride_b0_b, stride_b0_h, stride_b0_m, stride_b0_n,
    stride_out_b, stride_out_h, stride_out_m, stride_out_d,
    batch_size, num_heads, N_CTX, D_MODEL,
    P_SEQ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_m = (N_CTX + P_SEQ + BLOCK_M - 1) // BLOCK_M
    pid_batch = pid // (num_heads * num_blocks_m)
    remaining = pid % (num_heads * num_blocks_m)
    pid_head = remaining // num_blocks_m
    pid_m = remaining % num_blocks_m

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    off_batch = pid_batch * stride_q_b
    off_head = pid_head * stride_q_h

    q_offset = off_batch + off_head + offs_m[:, None] * stride_q_m + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_q_d
    q_mask = (offs_m[:, None] < (N_CTX + P_SEQ)) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL)
    q = tl.load(Q + q_offset, mask=q_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, N_CTX + P_SEQ, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        k_offset = off_batch + off_head + offs_n[None, :] * stride_k_m + tl.arange(0, BLOCK_DMODEL)[:, None] * stride_k_d
        k_mask = (offs_n[None, :] < (N_CTX + P_SEQ)) & (tl.arange(0, BLOCK_DMODEL)[:, None] < D_MODEL)
        k = tl.load(K + k_offset, mask=k_mask, other=0.0, eviction_policy="evict_first")

        qk = tl.dot(q, tl.trans(k.to(q.dtype))) * sm_scale
        
        b0_offset = off_batch + off_head + offs_m[:, None] * stride_b0_m + offs_n[None, :] * stride_b0_n
        b0_mask = (offs_m[:, None] < (N_CTX + P_SEQ)) & (offs_n[None, :] < (N_CTX + P_SEQ))
        b0 = tl.load(B0 + b0_offset, mask=b0_mask, other=0.0)
        
        scores = qk + b0
        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(scores - m_i_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_i_new = alpha * l_i + l_ij

        v_offset = off_batch + off_head + offs_n[:, None] * stride_v_m + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_v_d
        v_mask = (offs_n[:, None] < (N_CTX + P_SEQ)) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL)
        v = tl.load(V + v_offset, mask=v_mask, other=0.0, eviction_policy="evict_first")

        acc = (acc.to(v.dtype) * alpha[:, None]) + tl.dot(p.to(v.dtype), v)
        m_i, l_i = m_i_new, l_i_new

    acc = acc / l_i[:, None]
    out_offset = off_batch + off_head + offs_m[:, None] * stride_out_m + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_out_d
    out_mask = (offs_m[:, None] < (N_CTX + P_SEQ)) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL)
    tl.store(Out + out_offset, acc.to(Out.dtype.element_ty), mask=out_mask)

def _attention_rel_h_rel_w_kernel_aligned_device(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, B0: torch.Tensor, 
    output: torch.Tensor, sm_scale: float, P_SEQ: int
):
    assert Q.dtype in [torch.float16, torch.bfloat16], "Q must be float16 or bfloat16"
    assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"
    assert B0.shape == (Q.shape[0], Q.shape[1], Q.shape[2] + P_SEQ, Q.shape[2] + P_SEQ), "B0 shape mismatch"
    
    batch_size, num_heads, seq_len, D_MODEL = Q.shape
    BLOCK_M, BLOCK_N, BLOCK_DMODEL = 128, 128, 64  # Tunable parameters
    STAGES = 3 if D_MODEL <= 64 else 1
    
    grid = (batch_size * num_heads * ((seq_len + P_SEQ + BLOCK_M - 1) // BLOCK_M), 1, 1)
    
    _fwd_kernel_aligned[grid](
        Q, K, V, B0, output,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        B0.stride(0), B0.stride(1), B0.stride(2), B0.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, num_heads, seq_len, D_MODEL,
        P_SEQ=P_SEQ,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        STAGES=STAGES,
    )

# Example usage
batch_size, num_heads, seq_len, d_model = 4, 8, 1024, 128
Q = torch.randn(batch_size, num_heads, seq_len, d_model, dtype=torch.float16, device='cuda')
K = Q.clone()
V = Q.clone()
B0 = torch.randn(batch_size, num_heads, seq_len + 64, seq_len + 64, dtype=torch.float16, device='cuda')
output = torch.empty_like(Q)

_attention_rel_h_rel_w_kernel_aligned_device(Q, K, V, B0, output, sm_scale=0.5, P_SEQ=64)
