: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr,  
              N_CTX_P1: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    q = tl.load(Q + off_z * stride_qz + off_h * stride_qh + off_m[:, None] * stride_qm + off_n[None, :] * stride_qk)  
    Q_scale = tl.load(Q_scale + off_z * stride_qz + off_h * stride_qh) 
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh + off_n[None, :] * stride_kn  
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh + off_n[None, :] * stride_vk  
    k_scale_ptrs = K_scale + off_z * stride_kz + off_h * stride_kh + off_n * stride_kk 
    oz = off_z * stride_oz 
    oh = off_h * stride_oh 
    o_m = N_CTX - (start_m + 1) * BLOCK_M + tl.arange(0, BLOCK_M)  
    o_n = tl.arange(0, BLOCK_N)  
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0  
    q_scale = tl.load(Q_scale + off_z * stride_qz + off_h * stride_qh) 
    if STAGE & 1:  
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, k_ptrs, k_scale_ptrs, v_ptrs,  
                                         start_m, BLOCK_M, HEAD_DIM, BLOCK_N,  
                                         off_m, off_n, N_CTX,  
                                         STAGE=1, N_CTX_P1=N_CTX + 1)  
    if STAGE & 2:  
        tl.debug_barrier()  
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, k_ptrs, k_scale_ptrs, v_ptrs,  
                                         start_m, BLOCK_M, HEAD_DIM, BLOCK_N,  
                                         off_m, off_n, N_CTX,  
                                         STAGE=2, N_CTX_P1=N_CTX + 1)  
    acc = acc * (1.0 / l_i[:, None])
    m_i = m_i - tl.math.log2(l_i)
    acc = tl.where(o_m[:, None] < (N_CTX - off_n[None, :]), acc, 0)
    out_ptrs = Out + oz + off_h * stride_oh + o_m[:, None] * stride_om + o_n[None, :] * stride_on  
    tl.store(out_ptrs, acc, mask=o_m[:, None] < (N_CTX - off_n[None, :]))
    return

@torch.no_grad()
def context_attention_fwd_ppl(Q, K, V, Q_scale, K_scale, Out, 
                            max_input_len, 
                            stage=3, 
                            BLOCK_M=128, 
                            BLOCK_N=64, 
                            num_warps=8, 
                            num_stages=3):
    if K_scale is None:
        scale = K.shape[-1]**-0.5
        K_scale = V_scale = torch.ones_like(Q_scale) * scale
    Z, H, N_CTX, HEAD_DIM = *K.shape, Q.shape[-1]
    if Q_scale is None:
        Q_scale = torch.ones((Q.shape[0], Q.shape[1]), dtype=torch.float, device=Q.device)
    if K_scale is None:
        K_scale = V_scale = torch.ones((K.shape[0], K.shape[1]), dtype=torch.float, device=K.device)
    if stage == 3:
        num_stages = 3
        num_warps = 8
    elif stage == 2:
        num_stages = 4
        num_warps = 4
    elif stage == 1:
        num_stages = 3
        num_warps = 4
    else:
        raise ValueError(f'Invalid stage: {stage}')
    grid = (triton.cdiv(max_input_len, BLOCK_M), H * Z, 1)
    N_CTX_P1 = N_CTX + 1
    _attn_fwd[grid](Q, K, V, Q_scale, K_scale, Out,  
                    Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),  
                    K.stride(0), K.stride(1), K.stride(2), K.stride(3),  
                    V.stride(0), V.stride(1), V.stride(2), V.stride(3),  
                    Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),  
                    Z, H, N_CTX_P1,  
                    HEAD_DIM=HEAD_DIM,  
                    BLOCK_M=BLOCK_M,  
                    BLOCK_N=BLOCK_N,  
                    STAGE=stage,  
                    num_warps=num_warps,  
                    num_stages=num_stages)
    return
