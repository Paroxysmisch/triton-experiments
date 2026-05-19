import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale,  #
    L,  #
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  #
    BLOCK_M_TRANS: tl.constexpr, BLOCK_N_TRANS: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,  #
    M_OFFSET: tl.constexpr, N_OFFSET: tl.constexpr,  #
    BLOCK_M_LOOP: tl.constexpr, BLOCK_N_LOOP: tl.constexpr,  #
    Q_ROWMAJOR: tl.constexpr, K_ROWMAJOR: tl.constexpr,  #
    BIAS: tl.constexpr, IS_CAUSAL: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    BLOCK_D_TRANS: tl.constexpr,  #
    stride_qm, stride_qn, stride_qd,  #
    stride_km, stride_kn, stride_kn_d,  #
    stride_vm, stride_vn, stride_vd,  #
    stride_sm, stride_sn,  #
    stride_qk_scales,  #
    stride_l_qm, stride_l_qn, stride_l_qd,  #
    stride_l_km, stride_l_kn, stride_l_kn_d,  #
    stride_l_bm, stride_l_bn,  #
    stride_o_m, stride_o_n, stride_o_d,  #
    stride_d_m, stride_d_n, stride_d_d,  #
    stride_da_m, stride_da_n,  #
    qk_scales_ptr,  #
):
    """
    Attention forward kernel.
    """
    start_m = tl.program_id(0)
    off_m = start_m * BLOCK_M_LOOP + tl.arange(0, BLOCK_M_LOOP)
    off_n = tl.arange(0, BLOCK_N_LOOP)
    m_range = start_m * BLOCK_M + off_m * BLOCK_M_TRANS
    n_range = off_n * BLOCK_N_TRANS

    off_d = tl.arange(0, BLOCK_D_TRANS)

    q_ptrs = Q + (m_range[:, None] * stride_qm + n_range[None, :] * stride_qn + off_d[None, None] * stride_qd)
    k_ptrs = K + (m_range[:, None] * stride_km + n_range[None, :] * stride_kn + off_d[:, None] * stride_kn_d)
    v_ptrs = V + (m_range[:, None] * stride_vm + n_range[None, :] * stride_vn + off_d[None, None] * stride_vd)

    k_scale_ptrs = qk_scales_ptr + m_range * stride_qk_scales
    q_scale = tl.load(k_scale_ptrs)

    if STAGE == 1:
        acc = tl.zeros((BLOCK_M_TRANS, BLOCK_N_TRANS), dtype=tl.float32)
    elif STAGE == 2:
        acc = tl.zeros((BLOCK_M_TRANS, BLOCK_N_TRANS), dtype=tl.float32)
    elif STAGE == 3:
        acc = tl.zeros((BLOCK_M_TRANS, BLOCK_N_TRANS), dtype=tl.float32)

    offs_m_trans = tl.arange(0, BLOCK_M_TRANS)
    offs_n_trans = tl.arange(0, BLOCK_N_TRANS)

    l_offs_qm = tl.zeros((BLOCK_M_TRANS,), dtype=tl.int64)
    l_offs_qn = tl.zeros((BLOCK_N_TRANS,), dtype=tl.int64)
    l_offs_kn = tl.zeros((BLOCK_N_TRANS,), dtype=tl.int64)

    if STAGE == 1:
        if Q_ROWMAJOR:
            q = tl.load(q_ptrs, mask=n_range[None, :] < N, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=m_range[:, None] < M, other=0.0)
        if K_ROWMAJOR:
            k = tl.load(k_ptrs, mask=off_d[:, None] < D, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=off_d[None, :] < D, other=0.0)

        q = q * q_scale

        if BIAS:
            bias_ptrs = K + (m_range[:, None] * stride_km + off_n[None, :] * stride_kn + off_d[:, None] * stride_kn_d)
            bias = tl.load(bias_ptrs, mask=off_n[None, :] < N, other=0.0)
            k = k + bias

        if IS_CAUSAL:
            k = tl.where(n_range[None, :] >= m_range[:, None], k, 0.0)

        k_scale = tl.load(k_scale_ptrs)
        qk = tl.zeros((BLOCK_M_TRANS, BLOCK_N_TRANS), dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk *= k_scale[None, :]
        tl.store(q_ptrs, q.to(q_ptrs.dtype.element_ty), mask=n_range[None, :] < N)
        tl.store(k_ptrs, k.to(k_ptrs.dtype.element_ty), mask=off_d[None, :] < D)

        if L is not None:
            l_q_ptrs = L + (m_range[:, None] * stride_l_qm + n_range[None, :] * stride_l_qn + off_d[None, None] * stride_l_qd)
            l_q = tl.load(l_q_ptrs, mask=n_range[None, :] < N, other=0.0)
            l_offs_qm += m_range
            l_offs_qn += n_range
            tl.store(l_q_ptrs, l_q.to(l_q_ptrs.dtype.element_ty), mask=n_range[None, :] < N)

    elif STAGE == 2:
        if Q_ROWMAJOR:
            q = tl.load(q_ptrs, mask=off_d[None, :] < D, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=n_range[None, :] < N, other=0.0)
        if K_ROWMAJOR:
            k = tl.load(k_ptrs, mask=n_range[:, None] < N, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=m_range[None, :] < M, other=0.0)

        if BIAS:
            bias_ptrs = Q + (off_m[:, None] * stride_qm + n_
