import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_b0s, stride_b0h,
    N_CTX: tl.constexpr, P_SEQ: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    sm_scale: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize pointers
    q_ptrs = Q + (pid_b * stride_qbs + pid_h * stride_qh + 
                  offs_m[:, None] * stride_qd + offs_d[None, :])
    k_ptrs = K + (pid_b * stride_kbs + pid_h * stride_kh + 
                  offs_d[:, None] * stride_kd)
    v_ptrs = V + (pid_b * stride_vbs + pid_h * stride_vh + 
                  offs_d[None, :] * stride_vd)
    b0_ptrs = B0 + (pid_b * stride_b0s + pid_h * stride_b0h)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load Q block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Main loop
    for start_n in range(0, N_CTX + P_SEQ, BLOCK_N):
        # Load K block
        k = tl.load(k_ptrs + start_n * stride_kd,
                   mask=start_n + offs_n[None, :] < N_CTX + P_SEQ,
                   other=0.0)
        
        # Compute attention scores
        qk = tl.dot(q, k)
        qk = qk * sm_scale

        # Add relative position bias
        rel_idx = offs_m[:, None] - (start_n + offs_n[None, :])
        b0 = tl.load(b0_ptrs + rel_idx + P_SEQ,
                    mask=(rel_idx >= -P_SEQ) & (rel_idx < P_SEQ),
                    other=float("-inf"))
        qk = qk + b0

        # Compute softmax
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Update running max/sum
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # Load V block and update accumulator
        v = tl.load(v_ptrs + start_n * stride_vd,
                   mask=start_n + offs_n[:, None] < N_CTX + P_SEQ,
                   other=0.0)
        
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        acc += tl.dot(p, v)

        # Update running max/sum
        l_i = l_i_new
        m_i = m_i_new

    # Write output
    out_ptrs = Out + (pid_b * stride_obs + pid_h * stride_oh +
                      offs_m[:, None] * stride_od + offs_d[None, :])
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out, 
                                               N_CTX, P_SEQ, sm_scale,
                                               BLOCK_M=128, BLOCK_N=128, 
                                               BLOCK_DMODEL=64):
    # Check input shapes and types
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == BLOCK_DMODEL
    assert q.dtype == k.dtype == v.dtype
    OUT_DTYPE = q.dtype

    # Configure meta-parameters
    num_warps = 4 if BLOCK_DMODEL <= 64 else 8
    num_stages = 2

    # Launch kernel
    grid = (triton.cdiv(N_CTX, BLOCK_M), q.shape[1], q.shape[0])
    _fwd_kernel_aligned[grid](
        q, k, v, b0, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        b0.stride(0), b0.stride(1),
        N_CTX=N_CTX, P_SEQ=P_SEQ,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return out
