_qm, stride_qk,  
              stride_kz, stride_kh, stride_kn, stride_kk,  
              stride_vz, stride_vh, stride_vk, stride_vn,  
              stride_oz, stride_oh, stride_om, stride_on,  
              Z, H, N_CTX, HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr, BLOCK_Q: tl.constexpr):
    tl.static_assert(BLOCK_Q == 128)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    q_ptrs = Q + q_offset + (start_m * BLOCK_Q + tl.arange(0, BLOCK_Q)).to(tl.int64)[:, None] * stride_qm + tl.arange(0, 16)[None, :] * stride_qk
    v_ptrs = V + v_offset + (start_m * BLOCK_Q + tl.arange(0, BLOCK_Q)).to(tl.int64)[:, None] * stride_vk + tl.arange(0, 16)[None, :] * stride_vn
    o_ptrs = Out + o_offset + (start_m * BLOCK_Q + tl.arange(0, BLOCK_Q)).to(tl.int64)[:, None] * stride_om + tl.arange(0, 16)[None, :] * stride_on
    offs_m = start_m * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = tl.arange(0, 16)
    m_i = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_Q, 16], dtype=tl.float32)
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX) & ((tl.arange(0, 16)[None, :] < 96)[:, None]), other=0.0)
    q_scale = tl.load(Q_scale + off_hz)
    K_block_ptr = K + k_offset + (tl.arange(0, 16)).to(tl.int64) * stride_kn
    V_block_ptr = V + v_offset + (tl.arange(0, 16)).to(tl.int64) * stride_vk
    K_scale_ptr = K_scale + off_hz + tl.arange(0, 16)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_block_ptr, V_block_ptr, K_scale_ptr, q_scale, 
                                    start_m, BLOCK_M=16, BLOCK_N=16, HEAD_DIM=HEAD_DIM, 
                                    STAGE=STAGE, offs_m=offs_m, offs_n=offs_n, N_CTX=N_CTX)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = o_ptrs - stride_om + stride_on
    tl.store(m_ptrs, m_i[:, None])
    acc = acc.to(tl.float16)
    tl.store(o_ptrs, acc)
    return

def _attn_fwd_triton(q, k, v, q_scale, k_scale, head_dim=128, block_m=64, block_n=64, num_warps=4, num_stages=3, stage=1):
    batch, head, seq_len, head_dim = q.shape
    k_shape = (batch, head, seq_len, head_dim)
    v_shape = (batch, head, seq_len, head_dim)
    o = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, block_m), batch * head, 1)
    with torch.cuda.device(q.device.index):
        _attn_fwd[grid](q, k, v, q_scale, k_scale, o,  
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  
                        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  
                        batch, head, seq_len, HEAD_DIM=head_dim, 
                        BLOCK_M=block_m, BLOCK_N=block_n,  
                        STAGE=stage, BLOCK_Q=block_m,  
                        num_warps=num_warps, num_stages=num_stages)
    return o, grid

def attn_fwd(q, k, v, q_scale, k_scale, block_m=64, block_n=64, num_warps=4, num_stages=3):
    o, _ = _attn_fwd_triton(q, k, v, q_scale, k_scale, head_dim=128, block_m=block_m, block_n=block_n, num_warps=num_warps, num_stages=num_stages, stage=1)
    o, _ = _attn_fwd_triton(o, k, v, q_scale, k_scale, head_dim=16, block_m=64, block_n=64, num_warps=num_warps, num_stages=num_stages, stage=2)
    return o
