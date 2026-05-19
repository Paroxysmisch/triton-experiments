import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_aligned(
    Q_ptr, K_ptr, V_ptr, B0_ptr, Out_ptr,
    stride_qz, stride_qh, stride_qq,
    stride_kz, stride_kh, stride_kk,
    stride_vz, stride_vh, stride_vk,
    stride_b0z, stride_b0h, stride_b0k,
    stride_oz, stride_oh, stride_oq,
    N_CTX, P_SEQ,
    sm_scale,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)

    offs_m = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    # Base pointers
    Q_ptrs = Q_ptr + pid_batch * stride_qz + pid_head * stride_qh + offs_m[:, None] * stride_qq + offs_d[None, :] 
    K_ptrs = K_ptr + pid_batch * stride_kz + pid_head * stride_kh
    V_ptrs = V_ptr + pid_batch * stride_vz + pid_head * stride_vh
    B0_ptrs = B0_ptr + pid_batch * stride_b0z + pid_head * stride_b0h
    Out_ptrs = Out_ptr + pid_batch * stride_oz + pid_head * stride_oh + offs_m[:, None] * stride_oq + offs_d[None, :]

    q = tl.load(Q_ptrs, mask=(offs_m < (N_CTX+P_SEQ))[:, None] & (offs_d < BLOCK_DMODEL)[None, :], other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # Compute partial scaled-dot-product
    # We'll iterate over all columns in steps of BLOCK_N
    # for the K/V dimension.
    m_prev = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_M,), dtype=tl.float32)
    seq_len = N_CTX + P_SEQ

    # Each step processes a chunk of K and V
    for start_n in range(0, seq_len, BLOCK_N):
        k = tl.load(
            K_ptrs + (start_n + offs_n)[None, :] * stride_kk + offs_d[:, None],
            mask=((start_n + offs_n) < seq_len)[None, :] & (offs_d < BLOCK_DMODEL)[:, None],
            other=0.0
        )
        # [BLOCK_DMODEL, BLOCK_N] -> need to transpose for matmul
        k_t = tl.transpose(k)
        # compute Q*K^T
        score = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d in range(0, BLOCK_DMODEL):
            score += (q[:, d] * k_t[:, d])
        score = score * sm_scale

        # Add bias from B0
        b0_offs = (start_n + offs_n)
        b0 = tl.load(
            B0_ptrs + offs_m[:, None] * stride_b0k + b0_offs[None, :],
            mask=(offs_m[:, None] < seq_len) & (b0_offs[None, :] < seq_len),
            other=0.0
        )
        score += b0

        # Softmax block
        # -- merge with running total
        m_curr = tl.maximum(tl.max(score, 1), m_prev)
        score = tl.exp(score - m_curr[:, None])
        l_curr = tl.sum(score, 1)

        # renormalize previous exp
        ratio = tl.exp(m_prev - m_curr)
        l_new = l_curr + l_prev * ratio
        score = score * (1.0 / l_new[:, None])
        acc = acc * (ratio[:, None]) + tl.dot(score.to(Q_ptr.dtype.element_ty), tl.load(
            V_ptrs + (start_n + offs_n)[None, :] * stride_vk + offs_d[:, None],
            mask=((start_n + offs_n) < seq_len)[None, :] & (offs_d < BLOCK_DMODEL)[:, None],
            other=0.0
        ).to(Q_ptr.dtype.element_ty))

        l_prev = l_new
        m_prev = m_curr

    # Store results
    tl.store(
        Out_ptrs,
        acc.to(Out_ptr.dtype.element_ty),
        mask=(offs_m < (N_CTX+P_SEQ))[:, None] & (offs_d < BLOCK_DMODEL)[None, :]
    )


def _attention_rel_h_rel_w_kernel_aligned_device(
    Q, K, V, B0, sm_scale
):
    # Shapes
    B, H, N_CTX_plus_P_SEQ, D = Q.shape

    # Dtype check
    assert Q.dtype in [tl.float16, tl.bfloat16], "Q must be fp16 or bf16"
    assert K.dtype in [tl.float16, tl.bfloat16], "K must be fp16 or bf16"
    assert V.dtype in [tl.float16, tl.bfloat16], "V must be fp16 or bf16"
    assert B0.dtype in [tl.float16, tl.bfloat16], "B0 must be fp16 or bf16"

    # Create output
    import torch
    OUT_DTYPE = torch.float16 if Q.dtype == torch.float16 else torch.bfloat16
    Out = torch.empty_like(Q, dtype=OUT_DTYPE)

    # Strides
    stride_qz = Q.stride(0)
    stride_qh = Q.stride(1)
    stride_qq = Q.stride(2)
    stride_kz = K.stride(0)
    stride_kh = K.stride(1)
    stride_kk = K.stride(2)
    stride_vz = V.stride(0)
    stride_vh = V.stride(1)
    stride_vk = V.stride(2)
    stride_b0z = B0.stride(0)
    stride_b0h = B0.stride(1)
    stride_b0k = B0.stride(2)
    stride_oz = Out.stride(0)
    stride_oh = Out.stride(1)
    stride_oq = Out.stride(2)

    # Grid
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = D
    grid = (B, H, (N_CTX_plus_P_SEQ + BLOCK_M - 1) // BLOCK_M)

    _fwd_kernel_aligned[grid](
        Q, K, V, B0, Out,
        stride_qz, stride_qh, stride_qq,
        stride_kz, stride_kh, stride_kk,
        stride_vz, stride_vh, stride_vk,
        stride_b0z, stride_b0h, stride_b0k,
        stride_oz, stride_oh, stride_oq,
        N_CTX_plus_P_SEQ, 0,  # if there's an additional P_SEQ offset, place it here
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )
    return Out
