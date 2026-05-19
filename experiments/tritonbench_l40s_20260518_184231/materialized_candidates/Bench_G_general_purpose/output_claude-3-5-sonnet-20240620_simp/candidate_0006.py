import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_aligned(
    # Pointers to matrices
    Q, K, V, B0, Out,
    # Matrix dimensions
    batch_size, seqlen_q, seqlen_k, num_heads, head_dim,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_b0_h, stride_b0_m, stride_b0_n,
    stride_ob, stride_oh, stride_om,
    # Scale for attention scores
    scale,
    # Block pointers
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seqlen_q, BLOCK_M)
    num_pid_n = tl.cdiv(seqlen_k, BLOCK_N)
    num_pid_h = num_heads
    
    # Block position
    pid_m = pid // (num_pid_n * num_pid_h)
    pid_h = (pid % (num_pid_n * num_pid_h)) // num_pid_n
    pid_n = pid % num_pid_n
    
    # Block pointers
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    
    # Initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * 1
    k_ptrs = K + offs_n[:, None] * stride_kn + offs_d[None, :] * 1
    v_ptrs = V + offs_n[:, None] * stride_vn + offs_d[None, :] * 1
    
    # Load Q, K, V
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    
    # Compute attention scores
    s = tl.dot(q, tl.trans(k))
    s = s * scale
    
    # Load and add relative positional embeddings
    b0_ptr = B0 + pid_h * stride_b0_h + offs_m[:, None] * stride_b0_m + offs_n[None, :] * stride_b0_n
    b0 = tl.load(b0_ptr, mask=(offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k), other=0.0)
    s = s + b0
    
    # Compute attention weights
    p = tl.softmax(s, axis=1)
    
    # Compute output
    o = tl.dot(p, v)
    
    # Store output
    out_ptr = Out + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * 1
    tl.store(out_ptr, o, mask=offs_m[:, None] < seqlen_q)

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, scale):
    batch_size, num_heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Configure block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = head_dim
    
    # Configure grid
    grid = (triton.cdiv(seqlen_q, BLOCK_M) * triton.cdiv(seqlen_k, BLOCK_N) * num_heads,)
    
    # Launch kernel
    _fwd_kernel_aligned[grid](
        q, k, v, b0, output,
        batch_size, seqlen_q, seqlen_k, num_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        b0.stride(0), b0.stride(1), b0.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return output
