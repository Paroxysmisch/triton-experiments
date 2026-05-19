import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, 
    q_scale, K_ptr, K_scale_ptr, V_ptr, 
    start_m, start_n, 
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, 
    BLOCK_N: tl.constexpr, STAGE: tl.constexpr, 
    offs_m, offs_n, N_CTX: tl.constexpr,
    stride_kn, stride_kk, stride_vk, stride_vn
):
    # Load K and K_scale
    k_ptrs = K_ptr + start_n * stride_kn + offs_n[:, None] * stride_kk
    k = tl.load(k_ptrs, mask=offs_n[:, None] < (N_CTX - start_n), other=0.0)
    k_scale = tl.load(K_scale_ptr + start_n // BLOCK_N)
    
    # Compute QK
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk *= q_scale * k_scale
    
    # Causal masking
    offs_n_base = start_n + offs_n
    mask = (offs_m[:, None] >= offs_n_base[None, :])
    qk = tl.where(mask, qk, float('-inf'))
    
    # Compute m_ij and update m_i
    m_ij = tl.max(qk, axis=1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    
    # Compute p and update l_i
    p = tl.exp(qk - m_i_new[:, None])
    l_i_new = alpha * l_i + tl.sum(p, axis=1)
    
    # Load V
    v_ptrs = V_ptr + start_n * stride_vk + offs_n[:, None] * stride_vn
    v = tl.load(v_ptrs, mask=offs_n[:, None] < (N_CTX - start_n), other=0.0)
    
    # Update acc
    acc_scale = alpha[:, None] * l_i[:, None] / l_i_new[:, None]
    acc = acc * acc_scale + tl.dot(p.to(v.dtype), v) / l_i_new[:, None]
    
    return acc, l_i_new, m_i_new

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Pointers
    Q_ptr = Q + pid_z * stride_qz + pid_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    K_ptr = K + pid_z * stride_kz + pid_h * stride_kh
    V_ptr = V + pid_z * stride_vz + pid_h * stride_vh
    Out_ptr = Out + pid_z * stride_oz + pid_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_on)
    
    # Load Q
    q = tl.load(Q_ptr, mask=offs_m[:, None] < N_CTX, other=0.0)
    q_scale = tl.load(Q_scale + pid_z * H + pid_h)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Loop over K and V blocks
    start_n = 0
    max_n = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX) if STAGE == 1 else N_CTX
    for start_n in range(0, max_n, BLOCK_N):
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, q_scale, K_ptr, K_scale + pid_z * H + pid_h, V_ptr,
            start_m=pid_m * BLOCK_M, start_n=start_n,
            BLOCK_M=BLOCK_M, HEAD_DIM=HEAD_DIM, BLOCK_N=BLOCK_N, STAGE=STAGE,
            offs_m=offs_m, offs_n=offs_n, N_CTX=N_CTX,
            stride_kn=stride_kn, stride_kk=stride_kk,
            stride_vk=stride_vk, stride_vn=stride_vn
        )
    
    # Store output
    tl.store(Out_ptr, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

def forward(q, k, v, q_scale, k_scale):
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert q.is_cuda and k.is_cuda and v.is_cuda
    Z, H, N_CTX, HEAD_DIM = q.shape
    
    # Ensure contiguous tensors
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_scale = q_scale.contiguous().view(Z, H)
    k_scale = k_scale.contiguous().view(Z, H, -1)
    
    # Output tensor
    out = torch.empty_like(q)
    
    # Kernel configuration
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z, H)
    
    # Launch kernel
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX, HEAD_DIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, STAGE=1
    )
    return out
