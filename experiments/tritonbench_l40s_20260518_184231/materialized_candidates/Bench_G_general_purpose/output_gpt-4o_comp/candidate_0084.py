import triton
import triton.language as tl

# Triton kernel for scaled dot-product attention
@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_bz, stride_bh, stride_bm, stride_bd,
    stride_oz, stride_oh, stride_om, stride_od,
    sm_scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    
    # Compute batch and head indices
    batch_idx = pid_hz // stride_qh
    head_idx = pid_hz % stride_qh

    # Compute the starting indices for Q, K, V
    q_start = pid_m * BLOCK_M
    k_start = 0

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load Q block
    Q_block = tl.load(Q + batch_idx * stride_qz + head_idx * stride_qh + q_start * stride_qm, mask=True)
    
    for n in range(0, stride_kn, BLOCK_N):
        # Load K and V blocks
        K_block = tl.load(K + batch_idx * stride_kz + head_idx * stride_kh + n * stride_kn, mask=True)
        V_block = tl.load(V + batch_idx * stride_vz + head_idx * stride_vh + n * stride_vn, mask=True)
        
        # Compute dot product between Q and K
        qk = tl.dot(Q_block, K_block)
        
        # Scale and add bias
        qk = qk * sm_scale
        B_block = tl.load(B0 + batch_idx * stride_bz + head_idx * stride_bh + q_start * stride_bm + n * stride_bd, mask=True)
        qk += B_block
        
        # Apply softmax
        qk = tl.math.exp2(qk)
        
        # Accumulate the results
        acc += tl.dot(qk, V_block)
    
    # Write back the results to Out
    tl.store(Out + batch_idx * stride_oz + head_idx * stride_oh + q_start * stride_om, acc)

# Wrapper function to launch the Triton kernel
def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale, num_warps=4, num_stages=2):
    assert q.shape[0] == k.shape[0] == v.shape[0], "Batch size must match"
    assert q.shape[1] == k.shape[1] == v.shape[1], "Head size must match"
    assert q.shape[2] == k.shape[2] == v.shape[2], "Sequence length must match"
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = q.shape[-1]
    
    # Determine grid size
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])
    
    # Launch the kernel
    _fwd_kernel_aligned[grid](
        q, k, v, rel_h_w, q,  # Assuming Out is in-place with q for simplicity
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        rel_h_w.stride(0), rel_h_w.stride(1), rel_h_w.stride(2), rel_h_w.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        sm_scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps, num_stages=num_stages
    )
