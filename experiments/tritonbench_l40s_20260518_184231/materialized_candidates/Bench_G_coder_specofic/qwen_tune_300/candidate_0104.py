import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    # Compute qk
    if STAGE == 1:
        q = tl.load(q + offs_m * BLOCK_DMODEL + offs_n)
        q = (q * qk_scale).to(tl.float16)
        lo = 0
        hi = BLOCK_N
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = lo + BLOCK_M
    q = tl.broadcast_to(q, [BLOCK_M, BLOCK_DMODEL])
    q = tl.trans(q)
    K_block_ptr = K_block_ptr + lo * BLOCK_DMODEL
    V_block_ptr = V_block_ptr + lo * BLOCK_DMODEL
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr + start_n * BLOCK_DMODEL + offs_n)
        k = k.to(tl.float16)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale
            qk = tl.where(mask, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            qk -= m_ij
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 0)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr + start_n * BLOCK_DMODEL + offs_n)
        v = v.to(tl.float16)
        if STAGE == 2:
            acc = acc + tl.dot(p.to(tl.float16), v)
        else:
            p = p.to(tl.float16)
            acc = acc + tl.dot(p, v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return acc, l_i, m_i

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "STAGE": 1}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "STAGE": 2}, num_warps=4),
    ],
    key=["seq_len", "BLOCK_DMODEL"],
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    seq_len,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(seq_len, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float16 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(seq_len, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, seq_len),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(seq_len, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504
    q = tl.load(Q_block_ptr)
    if STAGE == 1:
        q = q.to(tl.float16)
    else:
        q = q.to(tl.float32)
    (
        acc,
        l_i,
        m_i,
    ) = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,
        start_m,
        qk_scale,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        STAGE,
        offs_m,
        offs_n,
    )
    if STAGE == 2:
        acc = acc / l_i[:, None]
        m_i = m_i + tl.math.log2(l_i)
    m_i += tl.math.log2(sm_scale)
    acc = acc.to(tl.float16)
    tl.store(O_block_ptr, acc, boundary_check=(0, 1))

def _attn_fwd_wrapper(
    q,
    k,
    v,
    o,
    sm_scale,
    max_input_len,
):
    grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
    pgm = _attn_fwd[grid](
        q,
        k,
        v,
        sm_scale,
        None,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
        max_input_len,
    )
    return pgm
