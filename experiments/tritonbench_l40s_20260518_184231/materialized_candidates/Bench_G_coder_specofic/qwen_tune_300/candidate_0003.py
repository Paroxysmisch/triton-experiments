import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,  # query, key, value, scale
    L,  # sequence length for padding
    Out,  # output
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lm,
    Z, H,  # batch and head dimensions
    N_CTX,  # number of context elements
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L + off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh,
        shape=(N_CTX, ),
        strides=(stride_lm, ),
        offsets=(start_m * BLOCK_M, ),
        block_shape=(BLOCK_M, ),
        order=(0, ),
    )
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504
    Q_ = tl.load(Q_block_ptr, boundary_check=(0, 1))
    if IS_TRITON_22:
        qk = tl.dot(Q_, K, allow_tf32=False)
    else:
        qk = tl.dot(Q_, K)
    qk = tl.math.mul(qk, qk_scale)
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    tl.store(L_block_ptr, m_i_new + acc_scale, boundary_check=(0, ))

    acc = tl.dot(p.to(Q_.dtype), V, allow_tf32=False)
    Out_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.store(Out_block_ptr, acc, boundary_check=(0, 1))


def context_attention_fwd(q, k, v, o, l, sm_scale, max_input_len=None):
    BLOCK = 128
    batch, head = q.shape[0], q.shape[1]
    grid = (triton.cdiv(q.shape[2], BLOCK), batch * head, 1)
    num_warps = 4 if BLOCK == 128 else 2

    if max_input_len is None:
        max_input_len = q.shape[2]

    def _run_fwd_kernel(
        q, k, v, sm_scale, l, o, stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh,
        stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on,
        stride_lz, stride_lh, stride_lm, Z, H, N_CTX, BLOCK_M, BLOCK_N,
        IS_TRITON_22, **meta
    ):
        _fwd_kernel[grid](
            q, k, v, sm_scale, l, o, stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh,
            stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on,
            stride_lz, stride_lh, stride_lm, Z, H, N_CTX, BLOCK_M, BLOCK_N,
            IS_TRITON_22=IS_TRITON_22, num_warps=num_warps, num_stages=1, **meta
        )

    if hasattr(triton, "version") and list(map(int, triton.__version__.split("."))) >= [
        2,
        2,
    ]:
        IS_TRITON_22 = True
    else:
        IS_TRITON_22 = False

    _run_fwd_kernel(
        q, k, v, sm_scale, l, o, q.stride(0), q.stride(1), q.stride(2),
        q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3), o.stride(0),
        o.stride(1), o.stride(2), o.stride(3), l.stride(0), l.stride(1),
        l.stride(2), batch, head, max_input_len, BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        IS_TRITON_22=IS_TRITON_22,
    )
