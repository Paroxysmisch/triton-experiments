import triton
import triton.language as tl
import math


@triton.jit
def fwd_decay_cumsum_kernel(
    g_ptr, g_o_ptr,
    decay,
    B, H, T, DK,
    stride_gBH, stride_gTH, stride_gDK,
    stride_goBH, stride_goTH, stride_goDK,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_DK: tl.constexpr
):
    pid_bh = tl.program_id(2)
    pid_t  = tl.program_id(1)
    pid_dk = tl.program_id(0)

    b = pid_bh // H
    h = pid_bh % H

    t_offset = pid_t * BLOCK_SIZE_T
    dk_offset = pid_dk * BLOCK_SIZE_DK

    t_range = t_offset + tl.arange(0, BLOCK_SIZE_T)
    dk_range = dk_offset + tl.arange(0, BLOCK_SIZE_DK)

    t_mask = t_range < T
    dk_mask = dk_range < DK
    mask = t_mask[:, None] & dk_mask[None, :]

    cum_decay = tl.zeros([BLOCK_SIZE_DK], dtype=tl.float32)

    for i in range(BLOCK_SIZE_T):
        ti = t_offset + i
        valid_t = ti < T
        if valid_t:
            g_index = b * stride_gBH + h * stride_gTH + ti * stride_gDK
            g_val = tl.load(g_ptr + g_index + dk_range, mask=dk_mask, other=0.0)
            # Example scaling by decay
            scaled_val = g_val * decay
            cum_decay = cum_decay + scaled_val

            go_index = b * stride_goBH + h * stride_goTH + ti * stride_goDK
            tl.store(g_o_ptr + go_index + dk_range, cum_decay, mask=dk_mask)


@triton.jit
def prepare_qg_kg_kernel(
    q_ptr, k_ptr, g_ptr,
    qg_ptr, kg_ptr,
    scale,
    B, H, T, DK,
    stride_qBH, stride_qTH, stride_qDK,
    stride_kBH, stride_kTH, stride_kDK,
    stride_gBH, stride_gTH, stride_gDK,
    stride_qgBH, stride_qgTH, stride_qgDK,
    stride_kgBH, stride_kgTH, stride_kgDK,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_DK: tl.constexpr
):
    pid_bh = tl.program_id(2)
    pid_t  = tl.program_id(1)
    pid_dk = tl.program_id(0)

    b = pid_bh // H
    h = pid_bh % H

    t_offset = pid_t * BLOCK_SIZE_T
    dk_offset = pid_dk * BLOCK_SIZE_DK

    t_range = t_offset + tl.arange(0, BLOCK_SIZE_T)
    dk_range = dk_offset + tl.arange(0, BLOCK_SIZE_DK)

    t_mask = t_range < T
    dk_mask = dk_range < DK
    mask = t_mask[:, None] & dk_mask[None, :]

    for i in range(BLOCK_SIZE_T):
        ti = t_offset + i
        valid_t = ti < T
        if valid_t:
            q_idx = b * stride_qBH + h * stride_qTH + ti * stride_qDK
            k_idx = b * stride_kBH + h * stride_kTH + ti * stride_kDK
            g_idx = b * stride_gBH + h * stride_gTH + ti * stride_gDK

            q_val = tl.load(q_ptr + q_idx + dk_range, mask=dk_mask, other=0.0)
            k_val = tl.load(k_ptr + k_idx + dk_range, mask=dk_mask, other=0.0)
            g_val = tl.load(g_ptr + g_idx + dk_range, mask=dk_mask, other=0.0)

            # Example exponential transform
            dec = tl.exp(-g_val * scale)
            q_transformed = q_val * dec
            k_transformed = k_val * dec

            qg_idx = b * stride_qgBH + h * stride_qgTH + ti * stride_qgDK
            kg_idx = b * stride_kgBH + h * stride_kgTH + ti * stride_kgDK

            tl.store(qg_ptr + qg_idx + dk_range, q_transformed, mask=dk_mask)
            tl.store(kg_ptr + kg_idx + dk_range, k_transformed, mask=dk_mask)


@triton.jit
def bwd_decay_global_cumsum_kernel(
    dq_inner_ptr, dq_inter_ptr,
    dk_inner_ptr, dk_inter_ptr,
    q_ptr, k_ptr, g_ptr, dg_ptr,
    scale,
    B, H, T, DK,
    stride_dqiBH, stride_dqiTH, stride_dqiDK,
    stride_dqjBH, stride_dqjTH, stride_dqjDK,
    stride_dkiBH, stride_dkiTH, stride_dkiDK,
    stride_dkjBH, stride_dkjTH, stride_dkjDK,
    stride_qBH, stride_qTH, stride_qDK,
    stride_kBH, stride_kTH, stride_kDK,
    stride_gBH, stride_gTH, stride_gDK,
    stride_dgBH, stride_dgTH, stride_dgDK,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_DK: tl.constexpr
):
    pid_bh = tl.program_id(2)
    pid_t  = tl.program_id(1)
    pid_dk = tl.program_id(0)

    b = pid_bh // H
    h = pid_bh % H

    t_offset = pid_t * BLOCK_SIZE_T
    dk_offset = pid_dk * BLOCK_SIZE_DK

    t_range = t_offset + tl.arange(0, BLOCK_SIZE_T)
    dk_range = dk_offset + tl.arange(0, BLOCK_SIZE_DK)

    t_mask = t_range < T
    dk_mask = dk_range < DK
    mask = t_mask[:, None] & dk_mask[None, :]

    # Simple backward step (reverse iteration)
    grad_accum = tl.zeros([BLOCK_SIZE_DK], dtype=tl.float32)

    for i in range(BLOCK_SIZE_T - 1, -1, -1):
        ti = t_offset + i
        valid_t = ti < T
        if valid_t:
            dqi_idx = b * stride_dqiBH + h * stride_dqiTH + ti * stride_dqiDK
            dqj_idx = b * stride_dqjBH + h * stride_dqjTH + ti * stride_dqjDK
            dki_idx = b * stride_dkiBH + h * stride_dkiTH + ti * stride_dkiDK
            dkj_idx = b * stride_dkjBH + h * stride_dkjTH + ti * stride_dkjDK

            q_idx   = b * stride_qBH +   h * stride_qTH   + ti * stride_qDK
            k_idx   = b * stride_kBH +   h * stride_kTH   + ti * stride_kDK
            g_idx   = b * stride_gBH +   h * stride_gTH   + ti * stride_gDK
            dg_idx  = b * stride_dgBH +  h * stride_dgTH  + ti * stride_dgDK

            dq_i = tl.load(dq_inner_ptr + dqi_idx + dk_range, mask=dk_mask, other=0.0)
            dq_j = tl.load(dq_inter_ptr + dqj_idx + dk_range, mask=dk_mask, other=0.0)
            dk_i = tl.load(dk_inner_ptr + dki_idx + dk_range, mask=dk_mask, other=0.0)
            dk_j = tl.load(dk_inter_ptr + dkj_idx + dk_range, mask=dk_mask, other=0.0)

            q_val  = tl.load(q_ptr + q_idx + dk_range, mask=dk_mask, other=0.0)
            k_val  = tl.load(k_ptr + k_idx + dk_range, mask=dk_mask, other=0.0)
            g_val  = tl.load(g_ptr + g_idx + dk_range, mask=dk_mask, other=0.0)

            # Example backward decay gradient
            sum_grad = dq_i + dq_j + dk_i + dk_j
            dec_factor = - scale * tl.exp(-g_val * scale)
            dg_val = sum_grad * (q_val + k_val) * dec_factor

            grad_accum = grad_accum + dg_val
            tl.store(dg_ptr + dg_idx + dk_range, grad_accum, mask=dk_mask)


def fwd_decay_cumsum(g, g_o, decay):
    B, H, T, DK = g.shape
    BLOCK_SIZE_T = 8
    BLOCK_SIZE_DK = 64

    grid_t = (T + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T
    grid_dk = (DK + BLOCK_SIZE_DK - 1) // BLOCK_SIZE_DK
    grid_bh = B * H

    stride_gBH  = H * T * DK
    stride_gTH  = T * DK
    stride_gDK  = DK
    stride_goBH = H * T * DK
    stride_goTH = T * DK
    stride_goDK = DK

    fwd_decay_cumsum_kernel[grid_dk, grid_t, grid_bh](
        g, g_o,
        decay,
        B, H, T, DK,
        stride_gBH, stride_gTH, stride_gDK,
        stride_goBH, stride_goTH, stride_goDK,
        BLOCK_SIZE_T=BLOCK_SIZE_T, BLOCK_SIZE_DK=BLOCK_SIZE_DK
    )


def prepare_qg_kg(q, k, g, qg, kg, scale):
    B, H, T, DK = q.shape
    BLOCK_SIZE_T = 8
    BLOCK_SIZE_DK = 64

    grid_t = (T + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T
    grid_dk = (DK + BLOCK_SIZE_DK - 1) // BLOCK_SIZE_DK
    grid_bh = B * H

    stride_qBH  = H * T * DK
    stride_qTH  = T * DK
    stride_qDK  = DK
    stride_kBH  = H * T * DK
    stride_kTH  = T * DK
    stride_kDK  = DK
    stride_gBH  = H * T * DK
    stride_gTH  = T * DK
    stride_gDK  = DK
    stride_qgBH = H * T * DK
    stride_qgTH = T * DK
    stride_qgDK = DK
    stride_kgBH = H * T * DK
    stride_kgTH = T * DK
    stride_kgDK = DK

    prepare_qg_kg_kernel[grid_dk, grid_t, grid_bh](
        q, k, g, qg, kg,
        scale,
        B, H, T, DK,
        stride_qBH, stride_qTH, stride_qDK,
        stride_kBH, stride_kTH, stride_kDK,
        stride_gBH, stride_gTH, stride_gDK,
        stride_qgBH, stride_qgTH, stride_qgDK,
        stride_kgBH, stride_kgTH, stride_kgDK,
        BLOCK_SIZE_T=BLOCK_SIZE_T, BLOCK_SIZE_DK=BLOCK_SIZE_DK
    )


def bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter,
                            q, k, g, dg, scale):
    B, H, T, DK = q.shape
    BLOCK_SIZE_T = 8
    BLOCK_SIZE_DK = 64

    grid_t = (T + BLOCK_SIZE_T - 1) // BLOCK_SIZE_T
    grid_dk = (DK + BLOCK_SIZE_DK - 1) // BLOCK_SIZE_DK
    grid_bh = B * H

    stride_dqiBH = H * T * DK
    stride_dqiTH = T * DK
    stride_dqiDK = DK
    stride_dqjBH = H * T * DK
    stride_dqjTH = T * DK
    stride_dqjDK = DK
    stride_dkiBH = H * T * DK
    stride_dkiTH = T * DK
    stride_dkiDK = DK
    stride_dkjBH = H * T * DK
    stride_dkjTH = T * DK
    stride_dkjDK = DK
    stride_qBH   = H *
