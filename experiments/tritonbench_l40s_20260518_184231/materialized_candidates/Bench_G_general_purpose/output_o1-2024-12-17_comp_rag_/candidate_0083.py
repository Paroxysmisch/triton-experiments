import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, N_CTX,
    BLOCK_M: tl.constexpr,  # block size in M dimension (sequence length)
    BLOCK_N: tl.constexpr,  # block size in N dimension (sequence length)
    BLOCK_DMODEL: tl.constexpr  # block size for dmodel dimension
):
    """Compute scaled dot-product attention for a block of Q, K, V with bias B0."""
    bid = tl.program_id(0)  # batch id
    hid = tl.program_id(1)  # head id
    start_m = tl.program_id(2)  # which block along the M dimension

    # Offsets in each tensor.
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # Q pointer
    q_ptrs = Q + bid * stride_qz + hid * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    # Output pointer
    out_ptrs = Out + bid * stride_oz + hid * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od

    # Load Q
    q_mask = offs_m < N_CTX
    q = tl.where(q_mask[:, None], tl.load(q_ptrs, mask=q_mask[:, None], other=0.0), 0.0)
    q = q.to(tl.float32)

    # Prepare accumulators for final computation
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    # Loop over blocks of columns (N dimension)
    num_blocks_n = (N_CTX + BLOCK_N - 1) // BLOCK_N
    for bn in range(num_blocks_n):
        offs_n = bn * BLOCK_N + tl.arange(0, BLOCK_N)
        # Load K and V
        k_ptrs = K + bid * stride_kz + hid * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = V + bid * stride_vz + hid * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k_mask = offs_n < N_CTX
        k = tl.where(k_mask[None, :], tl.load(k_ptrs, mask=k_mask[None, :], other=0.0), 0.0)
        k = k.to(tl.float32)
        v = tl.where(k_mask[:, None], tl.load(v_ptrs, mask=k_mask[:, None], other=0.0), 0.0)
        v = v.to(tl.float32)

        # QK^T
        qk = tl.dot(q, k)
        qk = qk * sm_scale

        # Add bias
        b0_ptrs = B0 + bid * stride_bz + hid * stride_bh + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
        b_mask = q_mask[:, None] & k_mask[None, :]
        bias_val = tl.where(b_mask, tl.load(b0_ptrs, mask=b_mask, other=0.0), 0.0)
        qk = qk + bias_val

        # Apply causal or validity mask if needed (here just ensure out-of-bound is -inf)
        # If you have actual causal or other masking logic, place it here.
        qk = tl.where((q_mask[:, None] & k_mask[None, :]), qk, float("-inf"))

        # Numerically stable softmax update per row
        m_ij = tl.max(qk, 1)
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)

        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        acc_scale = (l_i * alpha) / l_i_new
        acc = acc * acc_scale[:, None]

        # Update accumulation
        acc += tl.dot(p.to(v.dtype), v)

        # Update m_i, l_i
        l_i = l_i_new
        m_i = m_i_new

    # Write out
    out = acc / l_i[:, None]
    tl.store(out_ptrs, out, mask=q_mask[:, None])


def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out, sm_scale):
    """Set up and launch the _fwd_kernel_aligned Triton kernel."""
    # Validate shapes and dtypes (simplified; adapt as needed)
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert b0.is_contiguous()
    assert out.is_contiguous()
    assert q.shape == out.shape
    assert q.shape[:-1] == k.shape[:-1] == v.shape[:-1] == b0.shape[:-2]
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32)

    Z = q.shape[0]  # batch
    H = q.shape[1]  # heads
    N_CTX = q.shape[2]
    D_MODEL = q.shape[3]

    # Tiling/block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = D_MODEL

    # Create grid
    grid_z = Z
    grid_h = H
    grid_m = (N_CTX + BLOCK_M - 1) // BLOCK_M
    grid = (grid_z, grid_h, grid_m)

    # Choose warps/stages
    num_warps = 4
    num_stages = 2

    _fwd_kernel_aligned[grid](
        q, k, v, b0, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b0.stride(0), b0.stride(1), b0.stride(2), b0.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=num_stages
    )
