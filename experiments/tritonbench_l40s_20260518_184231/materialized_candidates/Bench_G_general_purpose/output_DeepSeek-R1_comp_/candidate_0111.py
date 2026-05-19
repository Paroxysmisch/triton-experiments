import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr, WINDOW_SIZE: tl.constexpr,
    qk_scale: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    Q_block = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    k_offset = off_z * stride_kz + off_h * stride_kh
    K_block = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )

    v_offset = off_z * stride_vz + off_h * stride_vh
    V_block = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    o_offset = off_z * stride_oz + off_h * stride_oh
    O_block = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_block, boundary_check=(0,))
    q = (q * qk_scale).to(tl.float16)

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(tl.advance(K_block, (0, start_n)), boundary_check=(1,))
        qk = tl.dot(q, k).to(tl.float32)
        
        if IS_CAUSAL or WINDOW_SIZE > 0:
            row = start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
            col = start_n + tl.arange(0, BLOCK_N)[None, :]
            mask = (col > row) if IS_CAUSAL else False
            if WINDOW_SIZE > 0:
                window_mask = tl.abs(row - col) > WINDOW_SIZE
                mask = mask | window_mask
            qk = tl.where(mask, float('-inf'), qk)

        m_curr = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.math.exp2(m_i - m_new)
        beta = tl.math.exp2(m_curr - m_new)
        l_curr = tl.sum(beta * tl.math.exp2(qk - m_new[:, None]), axis=1)
        l_i = l_i * alpha + l_curr
        
        p = beta[:, None] * tl.math.exp2(qk - m_new[:, None])
        p = p.to(V.dtype.element_ty)
        
        v = tl.load(tl.advance(V_block, (start_n, 0)), boundary_check=(0,))
        acc += tl.dot(p, v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O_block, acc.to(Out.dtype.element_ty), boundary_check=(0,))


def _forward(q, k, v, o, qk_scale, is_causal, window_size):
    BLOCK_M, BLOCK_N = 128, 64
    try:
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])
        _attn_fwd[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            IS_CAUSAL=is_causal, WINDOW_SIZE=window_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=q.shape[3], qk_scale=qk_scale
        )
    except triton.OutOfResources:
        BLOCK_M, BLOCK_N = 64, 32
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])
        _attn_fwd[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            IS_CAUSAL=is_causal, WINDOW_SIZE=window_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=q.shape[3], qk_scale=qk_scale
        )
