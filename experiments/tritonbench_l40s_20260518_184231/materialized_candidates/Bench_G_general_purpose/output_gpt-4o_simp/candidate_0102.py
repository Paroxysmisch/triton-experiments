import triton
import triton.language as tl

@triton.jit
def parallel_retention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    n_heads, seq_len, d_head, decay_factor,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Block pointers for Q, K, V, and output
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
    k_ptrs = K_ptr + offs_m[:, None] * stride_kn + offs_n[None, :] * stride_kk
    v_ptrs = V_ptr + offs_m[:, None] * stride_vn + offs_n[None, :] * stride_vk
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_ok

    # Load Q, K, V
    Q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len)
    K = tl.load(k_ptrs, mask=offs_n[None, :] < seq_len)
    V = tl.load(v_ptrs, mask=offs_n[None, :] < seq_len)

    # Compute scaled dot-product attention
    scores = tl.dot(Q, K.T) * decay_factor
    scores = tl.softmax(scores, axis=-1)
    output = tl.dot(scores, V)

    # Store output
    tl.store(out_ptrs, output, mask=offs_m[:, None] < seq_len)

@triton.jit
def parallel_retention_bwd_kernel(
    dOut_ptr, Q_ptr, K_ptr, V_ptr, dQ_ptr, dK_ptr, dV_ptr,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    n_heads, seq_len, d_head, decay_factor,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Block pointers for dOut, Q, K, V, and gradients
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dOut_ptrs = dOut_ptr + offs_m[:, None] * stride_dom + offs_n[None, :] * stride_dok
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
    k_ptrs = K_ptr + offs_m[:, None] * stride_kn + offs_n[None, :] * stride_kk
    v_ptrs = V_ptr + offs_m[:, None] * stride_vn + offs_n[None, :] * stride_vk
    dq_ptrs = dQ_ptr + offs_m[:, None] * stride_dqm + offs_n[None, :] * stride_dqk
    dk_ptrs = dK_ptr + offs_m[:, None] * stride_dkn + offs_n[None, :] * stride_dkk
    dv_ptrs = dV_ptr + offs_m[:, None] * stride_dvn + offs_n[None, :] * stride_dvk

    # Load dOut, Q, K, V
    dOut = tl.load(dOut_ptrs, mask=offs_m[:, None] < seq_len)
    Q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len)
    K = tl.load(k_ptrs, mask=offs_n[None, :] < seq_len)
    V = tl.load(v_ptrs, mask=offs_n[None, :] < seq_len)

    # Compute gradients
    dV = tl.dot(dOut.T, K)
    dK = tl.dot(Q.T, dOut)
    dQ = tl.dot(dOut, V.T)

    # Store gradients
    tl.store(dq_ptrs, dQ, mask=offs_m[:, None] < seq_len)
    tl.store(dk_ptrs, dK, mask=offs_n[None, :] < seq_len)
    tl.store(dv_ptrs, dV, mask=offs_n[None, :] < seq_len)

def forward(Q, K, V, decay_factor):
    # Allocate output tensor
    Out = torch.empty_like(Q)

    # Launch kernel
    grid = (triton.cdiv(Q.shape[0], BLOCK_SIZE_M), triton.cdiv(Q.shape[1], BLOCK_SIZE_N))
    parallel_retention_fwd_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Q.shape[1], Q.shape[2], Q.shape[3], decay_factor,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128
    )

    # Save tensors for backward pass
    ctx.save_for_backward(Q, K, V, Out)
    return Out

def backward(ctx, dOut):
    Q, K, V, Out = ctx.saved_tensors

    # Allocate gradient tensors
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    # Launch kernel
    grid = (triton.cdiv(dOut.shape[0], BLOCK_SIZE_M), triton.cdiv(dOut.shape[1], BLOCK_SIZE_N))
    parallel_retention_bwd_kernel[grid](
        dOut, Q, K, V, dQ, dK, dV,
        dOut.stride(0), dOut.stride(1), dOut.stride(2), dOut.stride(3),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        Q.shape[1], Q.shape[2], Q.shape[3], ctx.decay_factor,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128
    )

    return dQ, dK, dV
