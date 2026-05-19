import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q, k, g, A,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_A_h, s_A_t, s_A_d,
    H, T, K, scale,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr, STAGE: tl.constexpr,
    OFFSET_M: tl.constexpr, OFFSET_N: tl.constexpr,
):
    # Kernel to compute a sub-block of A using q, k, g
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_vo = tl.program_id(2)
    off_hz_vo = off_hz * VO + off_vo
    m_i = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_i = tl.arange(0, BLOCK_N)
    d_i = tl.arange(0, BLOCK_DMODEL)
    ha = tl.arange(0, STAGE)
    q_ptrs = q + (m_i[:, None] * s_qk_h + (off_hz + ha)[None, :] * s_qk_t + d_i[None, :] * s_qk_d)
    k_ptrs = k + (n_i[:, None] * s_qk_h + (off_hz + ha)[:, None] * s_qk_t + d_i[None, :] * s_qk_d)
    g_ptrs = g + ((m_i + OFFSET_M)[:, None] * s_vo_h + (off_hz_vo + OFFSET_N)[None, :] * s_vo_t + n_i[None, :] * s_vo_d)
    A_ptrs = A + (m_i[:, None] * s_A_h + n_i[None, :] * s_A_t)
    mask = (m_i[:, None] >= (OFFSET_M + start_m * BLOCK_M)) & (n_i[None, :] >= OFFSET_N)
    b_A = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for _ in range(K):
        b_q = tl.load(q_ptrs, mask=mask, other=0.0).to(tl.float32)
        b_k = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.float32)
        b_A += tl.dot(b_q, tl.trans(b_k))
        q_ptrs += BLOCK_DMODEL
        k_ptrs += BLOCK_DMODEL
    b_A *= scale
    b_g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    b_A = tl.exp(b_A) * b_g
    tl.store(A_ptrs, b_A.to(A.dtype.element_ty), mask=mask)

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q, k, g, A,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_A_h, s_A_t, s_A_d,
    H, T, K, scale,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Kernel to compute a sub-block of A using q, k, g
    off_hz = tl.program_id(0)
    off_vo = tl.program_id(1)
    off_hz_vo = off_hz * VO + off_vo
    m_i = tl.arange(0, BLOCK_M)
    d_i = tl.arange(0, BLOCK_DMODEL)
    ha = tl.arange(0, STAGE)
    q_ptrs = q + (m_i[:, None] * s_qk_h + (off_hz + ha)[None, :] * s_qk_t + d_i[None, :] * s_qk_d)
    k_ptrs = k + (m_i[:, None] * s_qk_h + (off_hz + ha)[None, :] * s_qk_t + d_i[None, :] * s_qk_d)
    g_ptrs = g + (m_i[:, None] * s_vo_h + (off_hz_vo)[None, :] * s_vo_t)
    A_ptrs = A + (m_i[:, None] * s_A_h)
    mask = (m_i[:, None] < (m_i[-1] + 1))
    b_A = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
    for i in range(K):
        b_q = tl.load(q_ptrs, mask=mask, other=0.0).to(tl.float32)
        b_k = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.float32)
        b_A += tl.dot(b_q, tl.trans(b_k))
        q_ptrs += BLOCK_DMODEL
        k_ptrs += BLOCK_DMODEL
    b_A *= scale
    b_g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    b_A = tl.exp(b_A) * b_g
    tl.store(A_ptrs, b_A.to(A.dtype.element_ty), mask=mask)

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q, k, g, A_intra,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_A_h, s_A_t, s_A_d,
    H, T, K, scale,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr, STAGE: tl.constexpr,
    OFFSET_M: tl.constexpr, OFFSET_N: tl.constexpr,
):
    # Kernel to compute a sub-block of A using q, k, g
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_vo = tl.program_id(2)
    off_hz_vo = off_hz * VO + off_vo
    m_i = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_i = tl.arange(0, BLOCK_N)
    d_i = tl.arange(0, BLOCK_DMODEL)
    ha = tl.arange(0, STAGE)
    q_ptrs = q + (m_i[:, None] * s_qk_h + (off_hz + ha)[None, :] * s_qk_t + d_i[None
