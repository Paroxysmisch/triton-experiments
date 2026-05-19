import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vm, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, H, N, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # program_id for batch-head combination
    bid = tl.program_id(0)
    # each block corresponds to a range of query positions
    m_block_id = tl.program_id(1)
    
    b_idx = bid // H
    h_idx = bid % H
    m_offset = m_block_id * BLOCK_M
    
    # Range of indices processed by this block in M dimension
    m_range = m_offset + tl.arange(0, BLOCK_M)
    # Range of indices in the head dimension
    d_range = tl.arange(0, BLOCK_D)
    
    # Pointers offset
    Q_block_ptr = Q_ptr + b_idx * stride_qb + h_idx * stride_qh + m_offset * stride_qm
    O_block_ptr = O_ptr + b_idx * stride_ob + h_idx * stride_oh + m_offset * stride_om
    
    # Load Q: shape [BLOCK_M, BLOCK_D]
    Qvals = tl.load(Q_block_ptr + (m_range[:, None] * stride_qm + d_range[None, :] * stride_qd), mask=(m_range[:, None] < N) & (d_range[None, :] < D), other=0.0)
    
    # We'll compute attention scores for all n in [0..N) in a loop
    # to avoid large memory usage, we split them in chunks of BLOCK_N
    # We'll accumulate partial results
    Out = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # Scaling factor for QK^T
    scale = 1.0 / tl.sqrt(float(D))
    
    for n_offset in range(0, N, BLOCK_N):
        n_range = n_offset + tl.arange(0, BLOCK_N)
        
        # Load K: [BLOCK_N, BLOCK_D]
        K_block_ptr = K_ptr + b_idx * stride_kb + h_idx * stride_kh + n_offset * stride_kn
        Kvals = tl.load(K_block_ptr + (n_range[:, None] * stride_kn + d_range[None, :] * stride_kd), mask=(n_range[:, None] < N) & (d_range[None, :] < D), other=0.0)
        
        # Compute attention scores = Q * K^T => shape [BLOCK_M, BLOCK_N]
        att_scores = tl.dot(Qvals, Kvals, trans_b=True) * scale
        
        # Row-wise softmax
        max_scores = tl.max(att_scores, 1)
        att_scores = att_scores - max_scores[:, None]
        exp_scores = tl.exp(att_scores)
        denom = tl.sum(exp_scores, 1)
        softmax_vals = exp_scores / denom[:, None]
        
        # Load V: [BLOCK_N, BLOCK_D] and multiply by attention
        V_block_ptr = V_ptr + b_idx * stride_vb + h_idx * stride_vh + n_offset * stride_vm
        Vvals = tl.load(V_block_ptr + (n_range[:, None] * stride_vm + d_range[None, :] * stride_vd), mask=(n_range[:, None] < N) & (d_range[None, :] < D), other=0.0)
        
        # Multiply softmax_vals [BLOCK_M, BLOCK_N] by Vvals [BLOCK_N, BLOCK_D]
        Out += tl.dot(softmax_vals, Vvals)
    
    # Store final results
    tl.store(
        O_block_ptr + (m_range[:, None] * stride_om + d_range[None, :] * stride_od), 
        Out, 
        mask=(m_range[:, None] < N) & (d_range[None, :] < D)
    )


def context_attention_fwd(Q, K, V, O, B, H, N, D):
    """
    Launches the Triton kernel '_fwd_kernel' to compute multi-head attention
    forward pass: O = softmax(QK^T / sqrt(D)) V
    
    Q, K, V, O are 4D tensors with shapes [B, H, N, D].
    B, H, N, D are batch, heads, sequence length, and head dimension sizes.
    """
    # Strides for Q
    stride_qb = Q.stride(0)
    stride_qh = Q.stride(1)
    stride_qm = Q.stride(2)
    stride_qd = Q.stride(3)
    # Strides for K
    stride_kb = K.stride(0)
    stride_kh = K.stride(1)
    stride_kn = K.stride(2)
    stride_kd = K.stride(3)
    # Strides for V
    stride_vb = V.stride(0)
    stride_vh = V.stride(1)
    stride_vm = V.stride(2)
    stride_vd = V.stride(3)
    # Strides for O
    stride_ob = O.stride(0)
    stride_oh = O.stride(1)
    stride_om = O.stride(2)
    stride_od = O.stride(3)
    
    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 64
    
    # Grid: x = B*H, y = ceil_div(N, BLOCK_M)
    grid = (B * H, (N + BLOCK_M - 1) // BLOCK_M)
    
    _fwd_kernel[grid](
        Q, K, V, O,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vm, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        B, H, N, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
    )
