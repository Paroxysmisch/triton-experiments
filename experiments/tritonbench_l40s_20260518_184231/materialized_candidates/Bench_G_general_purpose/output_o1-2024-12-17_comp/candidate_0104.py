import triton
import triton.language as tl
import torch

BLOCK_M = 64
BLOCK_N = 64
BLOCK_DMODEL = 64

@triton.jit
def _attn_fwd_inner(
    Q_PTR, K_PTR, V_PTR, O_PTR,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    q_scale, k_scale,
    n_ctx, d_model,
    OFF_Z, OFF_H, OFF_M
):
    pid_m = tl.program_id(0)
    m_block_start = pid_m * BLOCK_M
    q_off = OFF_Z * stride_qz + OFF_H * stride_qh + m_block_start * stride_qm
    o_off = OFF_Z * stride_oz + OFF_H * stride_oh + m_block_start * stride_om

    q_idx = m_block_start + tl.arange(0, BLOCK_M)
    mask_m = q_idx < n_ctx

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n_block_start in range(0, n_ctx, BLOCK_N):
        k_off = OFF_Z * stride_kz + OFF_H * stride_kh + n_block_start * stride_kn
        v_off = OFF_Z * stride_vz + OFF_H * stride_vh + n_block_start * stride_vn

        k_idx = n_block_start + tl.arange(0, BLOCK_N)
        mask_n = k_idx < n_ctx

        q = tl.load(Q_PTR + q_off + tl.arange(0, BLOCK_M)[:, None] * stride_qd, mask=mask_m[:, None], other=0.0)
        k = tl.load(K_PTR + k_off + tl.arange(0, BLOCK_N)[:, None] * stride_kd, mask=mask_n[:, None], other=0.0)
        q = q.to(tl.float32)
        k = k.to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * q_scale * k_scale
        new_m = tl.maximum(m_i[:, None], qk)
        alpha = tl.exp(m_i[:, None] - new_m)
        beta = tl.exp(qk - new_m)

        l_i = alpha * l_i + tl.sum(beta, 1)
        acc = alpha[:, None] * acc + tl.dot(beta, tl.load(V_PTR + v_off + tl.arange(0, BLOCK_N)[:, None] * stride_vd, 
                                                           mask=mask_n[:, None], other=0.0).to(tl.float32))
        m_i = new_m[:, 0]

    out = acc / l_i[:, None]
    tl.store(O_PTR + o_off + tl.arange(0, BLOCK_M)[:, None] * stride_od, out, mask=mask_m[:, None])


@triton.jit
def _attn_fwd(
    Q_PTR, K_PTR, V_PTR, O_PTR,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    q_scale, k_scale,
    n_ctx, d_model
):
    pid_z = tl.program_id(2)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(0)
    _attn_fwd_inner[1](
        Q_PTR, K_PTR, V_PTR, O_PTR,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_oz, stride_oh, stride_om, stride_od,
        q_scale, k_scale,
        n_ctx, d_model,
        pid_z, pid_h, pid_m
    )

def forward(q, k, v, q_scale, k_scale):
    assert q.shape == k.shape == v.shape
    bsz, heads, seq_len, dim = q.shape
    o = torch.empty_like(q)

    q_ptr = q
    k_ptr = k
    v_ptr = v
    o_ptr = o

    grid = ( (seq_len + BLOCK_M - 1)//BLOCK_M, heads, bsz )

    triton.run(
        _attn_fwd,
        grid=grid,
        args=[
            q_ptr, k_ptr, v_ptr, o_ptr,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q_scale, k_scale,
            seq_len, dim
        ],
        num_warps=4
    )
    return o
