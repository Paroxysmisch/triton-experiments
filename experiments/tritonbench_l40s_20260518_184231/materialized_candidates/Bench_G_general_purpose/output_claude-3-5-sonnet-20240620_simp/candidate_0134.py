import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    # Pointers to matrices
    Q, K, V, Out,
    # Matrix dimensions
    batch_size, num_heads, seqlen_q, seqlen_k, head_dim,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seqlen_q, BLOCK_M)
    num_pid_n = tl.cdiv(seqlen_k, BLOCK_N)
    num_pid_in_group = num_pid_n * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, num_pid_m - first_pid_m)
    pid_m = (pid % group_size_m) + first_pid_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pointers for Q, K, V blocks
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + (offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk)
    
    # Load Q, K blocks
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    k = tl.load(k_ptrs, mask=offs_n[None, :] < seqlen_k, other=0.0)
    
    # Compute attention scores
    scale = 1.0 / tl.sqrt(float(head_dim))
    scores = tl.dot(q, k) * scale
    
    # Compute softmax
    scores = scores - tl.max(scores, 1)[:, None]
    scores = tl.exp(scores)
    normalizer = tl.sum(scores, 1)[:, None]
    scores = scores / normalizer
    
    # Load V block and compute weighted sum
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    acc = tl.dot(scores, v)
    
    # Write output
    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_ptrs = Out + (offs_m_out[:, None] * stride_om + offs_k[None, :] * stride_ok)
    tl.store(out_ptrs, acc, mask=offs_m_out[:, None] < seqlen_q)

def context_attention_fwd(q, k, v):
    """
    Compute attention: softmax(Q @ K.T) @ V
    """
    # Shape constraints
    batch_size, num_heads, seqlen_q, head_dim = q.shape
    _, _, seqlen_k, _ = k.shape
    assert k.shape == (batch_size, num_heads, seqlen_k, head_dim)
    assert v.shape == (batch_size, num_heads, seqlen_k, head_dim)
    
    # Output tensor
    out = torch.empty_like(q)
    
    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = head_dim
    
    # Grid size
    grid = (triton.cdiv(seqlen_q, BLOCK_M) * triton.cdiv(seqlen_k, BLOCK_N),)
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, out,
        batch_size, num_heads, seqlen_q, seqlen_k, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
    )
    
    return out
