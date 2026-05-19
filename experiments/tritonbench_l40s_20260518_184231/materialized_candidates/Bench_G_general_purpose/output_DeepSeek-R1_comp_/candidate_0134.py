import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    B_Start_Loc, B_Seqlen,
    Out,
    stride_qz, stride_qh, stride_qd,
    stride_kz, stride_kh, stride_kd,
    stride_vz, stride_vh, stride_vd,
    stride_oz, stride_oh, stride_od,
    head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Get program indices
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Get batch-specific metadata
    start_loc = tl.load(B_Start_Loc + pid_batch)
    seq_len = tl.load(B_Seqlen + pid_batch)
    
    # Calculate valid range for current block
    q_offset = pid_m * BLOCK_M
    if q_offset >= seq_len:
        return
    
    # Initialize pointers for current Q block
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    Q_ptr = Q + (start_loc + q_offset + offs_m[:, None]) * stride_qz + \
            pid_head * stride_qh + offs_d[None, :] * stride_qd
    valid_q = (q_offset + offs_m) < seq_len
    q = tl.load(Q_ptr, mask=valid_q[:, None] & (offs_d[None, :] < head_dim), other=0.0)
    
    # Initialize accumulation registers
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Loop over K/V blocks
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    for k_off in range(0, num_blocks * BLOCK_N, BLOCK_N):
        # Load K block
        offs_n = tl.arange(0, BLOCK_N)
        K_ptr = K + (start_loc + k_off + offs_n[:, None]) * stride_kz + \
                pid_head * stride_kh + offs_d[None, :] * stride_kd
        valid_k = (k_off + offs_n) < seq_len
        k = tl.load(K_ptr, mask=valid_k[:, None] & (offs_d[None, :] < head_dim), other=0.0)
        
        # Load V block
        V_ptr = V + (start_loc + k_off + offs_n[:, None]) * stride_vz + \
                pid_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V_ptr, mask=valid_k[:, None] & (offs_d[None, :] < head_dim), other=0.0)
        
        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(valid_q[:, None] & valid_k[None, :], qk, float("-inf"))
        qk *= sm_scale
        
        # Compute attention updates
        m_curr = tl.maximum(tl.max(qk, 1), m_i)
        alpha = tl.exp(qk - m_curr[:, None])
        l_curr = tl.exp(m_i - m_curr) * l_i + tl.sum(alpha, 1)
        
        # Update accumulators
        p = alpha.to(v.dtype)
        acc_scale = tl.exp(m_i[:, None] - m_curr[:, None])
        acc = acc * acc_scale + tl.dot(p, v)
        
        # Maintain max and sum
        m_i = m_curr
        l_i = l_curr
    
    # Finalize output
    acc = acc / l_i[:, None]
    
    # Write output
    Out_ptr = Out + (start_loc + q_offset + offs_m[:, None]) * stride_oz + \
              pid_head * stride_oh + offs_d[None, :] * stride_od
    tl.store(Out_ptr, acc.to(Out_ptr.dtype.element_ty), 
             mask=valid_q[:, None] & (offs_d[None, :] < head_dim))

def context_attention_fwd(q, k, v, o, B_Start_Loc, B_Seqlen, 
                          stride_qz, stride_qh, stride_qd,
                          stride_kz, stride_kh, stride_kd,
                          stride_vz, stride_vh, stride_vd,
                          stride_oz, stride_oh, stride_od,
                          head_dim, sm_scale=None):
    # Handle default scale
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Determine kernel parameters
    max_seq_len = torch.max(B_Seqlen).item()
    BLOCK_M = 128
    BLOCK_N = 64 if head_dim <= 64 else 128
    BLOCK_DMODEL = head_dim
    
    # Configure grid
    grid = (B_Start_Loc.shape[0], k.shape[1], triton.cdiv(max_seq_len, BLOCK_M))
    
    # Tune warps based on block size
    num_warps = 4
    if BLOCK_N >= 128:
        num_warps = 8
    elif BLOCK_N >= 256:
        num_warps = 16
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, sm_scale,
        B_Start_Loc, B_Seqlen, o,
        stride_qz, stride_qh, stride_qd,
        stride_kz, stride_kh, stride_kd,
        stride_vz, stride_vh, stride_vd,
        stride_oz, stride_oh, stride_od,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=4,
    )
