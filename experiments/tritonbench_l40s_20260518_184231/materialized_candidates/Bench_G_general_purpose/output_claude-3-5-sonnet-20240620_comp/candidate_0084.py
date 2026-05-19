import torch
import triton
import triton.language as tl

# Define block sizes for efficient memory access
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

@triton.jit
def _fwd_kernel_aligned(
    # Pointers to matrices
    Q, K, V, B0, Out,
    # Matrix dimensions
    H, N_CTX,
    # Strides for accessing different dimensions
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_b0_h, stride_b0_m, stride_b0_n,
    stride_om, stride_oh, stride_on,
    # Scale for attention scores
    sm_scale,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize pointers
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd + pid_h * stride_qh)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd + pid_h * stride_kh)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd + pid_h * stride_vh)
    
    # Load Q, K blocks
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
    
    # Compute attention scores
    qk = tl.dot(q, k.transpose())
    qk = qk * sm_scale

    # Add relative position bias
    b0_ptrs = B0 + (pid_h * stride_b0_h + offs_m[:, None] * stride_b0_m + offs_n[None, :] * stride_b0_n)
    bias = tl.load(b0_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX), other=0.0)
    qk = qk + bias

    # Compute softmax
    qk_max = tl.max(qk, 1)
    qk_exp = tl.exp2(qk - qk_max[:, None])
    qk_sum = tl.sum(qk_exp, 1)
    softmax = qk_exp / qk_sum[:, None]

    # Load V block and compute output
    v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
    out = tl.dot(softmax, v)

    # Write output
    out_ptrs = Out + (offs_m[:, None] * stride_om + offs_d[None, :] * 1 + pid_h * stride_oh)
    tl.store(out_ptrs, out, mask=offs_m[:, None] < N_CTX)

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale):
    batch_size, n_heads, seq_len, d_head = q.shape
    
    # Check input shapes
    assert k.shape == (batch_size, n_heads, seq_len, d_head)
    assert v.shape == (batch_size, n_heads, seq_len, d_head)
    assert rel_h_w.shape == (n_heads, seq_len, seq_len)
    
    # Output tensor
    output = torch.empty_like(q)
    
    # Configure grid
    grid = (
        triton.cdiv(seq_len, BLOCK_M),
        triton.cdiv(seq_len, BLOCK_N),
        batch_size * n_heads
    )
    
    # Launch kernel
    _fwd_kernel_aligned[grid](
        q, k, v, rel_h_w, output,
        n_heads, seq_len,
        q.stride(2), q.stride(1), q.stride(3),
        k.stride(2), k.stride(1), k.stride(3),
        v.stride(2), v.stride(1), v.stride(3),
        rel_h_w.stride(0), rel_h_w.stride(1), rel_h_w.stride(2),
        output.stride(2), output.stride(1), output.stride(3),
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2
    )
    
    return output
