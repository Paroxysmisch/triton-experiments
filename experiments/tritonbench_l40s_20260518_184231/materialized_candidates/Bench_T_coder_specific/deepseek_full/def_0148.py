import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@triton.jit
def adam_kernel(
    p_ptr,
    m_ptr,
    v_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    wd_ratio,
    wd_type,
    do_wd,
    t,
    betas_t,
    wd_const_ptr,
    group_size,
    block_size,
    use_amsgrad,
    max_opt_level,
    maximize,
    fp32_inv_scale,
    adam8_bit_update,
    adam8_bit_scale_ptr,
    adam8_bit_zeros_ptr,
    adam8_bit_radius_ptr,
    adam8_bit_min_ptr,
    adam8_bit_max_ptr,
    adam8_bit_qmin_ptr,
    adam8_bit_qmax_ptr,
    BLOCK_SIZE: tl.constexpr,
    D_P: tl.constexpr,
    D_M: tl.constexpr,
    D_V: tl.constexpr,
    D_WD: tl.constexpr,
    D_SCALE: tl.constexpr,
):
    row_block_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    m_ptr += group_idx * D_M
    v_ptr += group_idx * D_V
    wd_const_ptr += group_idx * D_WD
    if D_SCALE:
        adam8_bit_scale_ptr += group_idx * D_SCALE
        adam8_bit_zeros_ptr += group_idx * D_SCALE
        adam8_bit_radius_ptr += group_idx * D_SCALE
        adam8_bit_min_ptr += group_idx * D_SCALE
        adam8_bit_max_ptr += group_idx * D_SCALE
        adam8_bit_qmin_ptr += group_idx * D_SCALE
        adam8_bit_qmax_ptr += group_idx * D_SCALE

    t_t = t - tl.arange(0, BLOCK_SIZE)
    betas_t = betas_t.to(tl.float32)
    beta1_t = tl.power(betas_t[0], t_t)
    beta2_t = tl.power(betas_t[1], t_t)
    p_ptr += row_block_idx * BLOCK_SIZE * D_P
    m_row_ptr = m_ptr + row_block_idx * BLOCK_SIZE
    v_row_ptr = v_ptr + row_block_idx * BLOCK_SIZE

    col_block_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_block_idx < group_size

    m = tl.load(m_row_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_row_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
    if D_SCALE:
        scale = tl.load(adam8_bit_scale_ptr + col_block_idx, mask=mask, other=1.0).to(tl.float32)
        zeros = tl.load(adam8_bit_zeros_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
        radius = tl.load(adam8_bit_radius_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
        min = tl.load(adam8_bit_min_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
        max = tl.load(adam8_bit_max_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
        qmin = tl.load(adam8_bit_qmin_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
        qmax = tl.load(adam8_bit_qmax_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)

    i = tl.load(wd_const_ptr + col_block_idx, mask=mask, other=0.0).to(tl.int32)

    if do_wd:
        if wd_type == "norm":
            if maximize:
                p = -tl.load(p_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
            else:
                p = tl.load(p_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
            p = p * (1 - wd_ratio)
            if i == 0:
                m = m * (1 - wd_ratio) + p * beta1_t
                v = v * (1 - wd_ratio) + p * p * beta2_t
            else:
                m = m * (1 - wd_ratio) + p * beta1_t
                v = v * (1 - wd_ratio) + p * p * beta2_t
        elif wd_type == "wd_const":
            if i == 0:
                m = m * (1 - wd_ratio)
                v = v * (1 - wd_ratio)
            else:
                if maximize:
                    p = -tl.load(p_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
                else:
                    p = tl.load(p_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
                m = m * (1 - wd_ratio) + p * beta1_t
                v = v * (1 - wd_ratio) + p * p * beta2_t

    if adam8_bit_update:
        if maximize:
            p = -tl.load(p_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
        else:
            p = tl.load(p_ptr + col_block_idx, mask=mask, other=0.0).to(tl.float32)
        m_hat = m * beta1_t
        denom = tl.sqrt(v * beta2_t + eps)
        p_8bit = triton_helpers.fp8_quant(p, scale, zeros, radius, min, max, qmin, qmax)
        update_8bit = m_hat / denom
        update_8bit = triton_helpers.fp8_quant(update_8bit, scale, zeros, radius, min, max, qmin, qmax)
        update_8bit = update_8bit - p_8bit
        update_8bit = triton_helpers.fp8_dequant(update_8bit, scale, zeros, radius, min, max, qmin, qmax)
        update_8bit = update_8bit * fp32_inv_scale
        if i == 0:
            m = m * (1 - wd_ratio) + p * beta1_t
            v = v * (1 - wd_ratio) + p * p * beta2_t
            tl.store(p_ptr + col_block_idx, update_8bit, mask=mask)
        else:
            m = m * (1 - wd_ratio) + update_8bit * beta1_t
            v = v * (1 - wd_ratio
