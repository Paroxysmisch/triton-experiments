import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, q_scale, k_scale, 
    Out, 
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
    off_k = off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_m[None, :] * stride_kk
    off_v = off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_m[None, :] * stride_vn
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    t_ptrs = tl.arange(0, BLOCK_M)
    # mask
    mask_m = tl.where(
        tl.arange(0, BLOCK_M) < start_m * BLOCK_M + BLOCK_M,
        tl.where(
            tl.arange(0, BLOCK_M) < start_m * BLOCK_M,
            tl.zeros((BLOCK_M,), dtype=tl.int32),
            1,
        ),
        0,
    )
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * q_scale).to(tl.float16)
    # loop over k, v and update accumulator
    for start_n in range(0, N_CTX, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_ptrs, mask=mask_m[None, :], other=0.0)
        k = (k * k_scale).to(tl.float16)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        # -- compute m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(qk - m_i_new[:, None])
        l_i_num = tl.sum(alpha * beta, 1)
        # -- update m_i and l_i
        l_i = l_i * tl.exp(m_i - m_i_new) + l_i_num
        m_i = m_i_new
        # -- update acc --
        acc *= (l_i[:, None] * alpha).to(tl.float16)
        # -- update qk --
        qk = (qk - m_i_new[:, None]).to(tl.float16)
        acc += tl.dot(tl.exp(qk), k)
        # update pointers to K, V
        k_ptrs += BLOCK_N * stride_kk
        v_ptrs += BLOCK_N * stride_vk
    # initialize pointers to output
    off_hz = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)

@triton.jit
def _attn_fwd_inner(
    Q, K, V, sm_scale, q_scale, k_scale, 
    Out, 
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
    off_k = off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_m[None, :] * stride_kk
    off_v = off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_m[None, :] * stride_vn
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    t_ptrs = tl.arange(0, BLOCK_M)
    # mask
    mask_m = tl.where(
        tl.arange(0, BLOCK_M) < start_m * BLOCK_M + BLOCK_M,
        tl.where(
            tl.arange(0, BLOCK_M) < start_m * BLOCK_M,
            tl.zeros((BLOCK_M,), dtype=tl.int32),
            1,
        ),
        0,
    )
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs,
