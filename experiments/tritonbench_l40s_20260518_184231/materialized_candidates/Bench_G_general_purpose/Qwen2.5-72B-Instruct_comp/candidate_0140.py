import triton
import triton.language as tl

@triton.jit
def fwd_decay_cumsum(
    g_ptr, g_o_ptr, decay, BT, DK, T, stride_g_d, stride_g_t, stride_g_o_d, stride_g_o_t,
    BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr
):
    pid_d = tl.program_id(axis=0)
    pid_t = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    block_start_d = pid_d * BLOCK_D
    block_start_t = pid_t * BLOCK_T

    mask = block_start_d + tl.arange(0, BLOCK_D) < DK
    mask &= block_start_t + tl.arange(0, BLOCK_T) < T

    cum_decay = tl.zeros((BLOCK_D,), dtype=tl.float32)
    inv_ln2 = 1.0 / tl.log(2.0)

    for i in range(BT):
        g_offsets = block_start_d + tl.arange(0, BLOCK_D)
        g_offsets = g_offsets + (block_start_t + i) * stride_g_t
        g = tl.load(g_ptr + g_offsets, mask=mask, other=0.0)

        decay_factor = tl.exp(cum_decay * inv_ln2)
        g = g * decay_factor

        cum_decay += g
        g_o_offsets = block_start_d + tl.arange(0, BLOCK_D)
        g_o_offsets = g_o_offsets + (block_start_t + i) * stride_g_o_t
        tl.store(g_o_ptr + g_o_offsets, g, mask=mask)

@triton.jit
def prepare_qg_kg(
    q_ptr, k_ptr, g_ptr, qg_ptr, kg_ptr, BT, DK, T, stride_q_d, stride_q_t, stride_k_d, stride_k_t,
    stride_g_d, stride_g_t, stride_qg_d, stride_qg_t, stride_kg_d, stride_kg_t,
    BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr
):
    pid_d = tl.program_id(axis=0)
    pid_t = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    block_start_d = pid_d * BLOCK_D
    block_start_t = pid_t * BLOCK_T

    mask = block_start_d + tl.arange(0, BLOCK_D) < DK
    mask &= block_start_t + tl.arange(0, BLOCK_T) < T

    cum_decay = tl.zeros((BLOCK_D,), dtype=tl.float32)
    inv_ln2 = 1.0 / tl.log(2.0)

    for i in range(BT):
        g_offsets = block_start_d + tl.arange(0, BLOCK_D)
        g_offsets = g_offsets + (block_start_t + i) * stride_g_t
        g = tl.load(g_ptr + g_offsets, mask=mask, other=0.0)

        decay_factor = tl.exp(cum_decay * inv_ln2)
        g = g * decay_factor

        cum_decay += g

        q_offsets = block_start_d + tl.arange(0, BLOCK_D)
        q_offsets = q_offsets + (block_start_t + i) * stride_q_t
        q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)
        q = q * decay_factor

        k_offsets = block_start_d + tl.arange(0, BLOCK_D)
        k_offsets = k_offsets + (block_start_t + i) * stride_k_t
        k = tl.load(k_ptr + k_offsets, mask=mask, other=0.0)
        k = k * decay_factor

        qg_offsets = block_start_d + tl.arange(0, BLOCK_D)
        qg_offsets = qg_offsets + (block_start_t + i) * stride_qg_t
        tl.store(qg_ptr + qg_offsets, q, mask=mask)

        kg_offsets = block_start_d + tl.arange(0, BLOCK_D)
        kg_offsets = kg_offsets + (block_start_t + i) * stride_kg_t
        tl.store(kg_ptr + kg_offsets, k, mask=mask)

@triton.jit
def bwd_decay_global_cumsum(
    dq_inner_ptr, dq_inter_ptr, dk_inner_ptr, dk_inter_ptr, q_ptr, k_ptr, g_ptr, dg_ptr, BT, DK, T,
    stride_dq_inner_d, stride_dq_inner_t, stride_dq_inter_d, stride_dq_inter_t, stride_dk_inner_d, stride_dk_inner_t,
    stride_dk_inter_d, stride_dk_inter_t, stride_q_d, stride_q_t, stride_k_d, stride_k_t, stride_g_d, stride_g_t,
    stride_dg_d, stride_dg_t, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr
):
    pid_d = tl.program_id(axis=0)
    pid_t = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    block_start_d = pid_d * BLOCK_D
    block_start_t = pid_t * BLOCK_T

    mask = block_start_d + tl.arange(0, BLOCK_D) < DK
    mask &= block_start_t + tl.arange(0, BLOCK_T) < T

    cum_decay = tl.zeros((BLOCK_D,), dtype=tl.float32)
    inv_ln2 = 1.0 / tl.log(2.0)

    for i in range(BT):
        g_offsets = block_start_d + tl.arange(0, BLOCK_D)
        g_offsets = g_offsets + (block_start_t + i) * stride_g_t
        g = tl.load(g_ptr + g_offsets, mask=mask, other=0.0)

        decay_factor = tl.exp(cum_decay * inv_ln2)
        g = g * decay_factor

        cum_decay += g

        q_offsets = block_start_d + tl.arange(0, BLOCK_D)
        q_offsets = q_offsets + (block_start_t + i) * stride_q_t
        q = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)

        k_offsets = block_start_d + tl.arange(0, BLOCK_D)
        k_offsets = k_offsets + (block_start_t + i) * stride_k_t
        k = tl.load(k_ptr + k_offsets, mask=mask, other=0.0)

        dq_inner_offsets = block_start_d + tl.arange(0, BLOCK_D)
        dq_inner_offsets = dq_inner_offsets + (block_start_t + i) * stride_dq_inner_t
        dq_inner = tl.load(dq_inner_ptr + dq_inner_offsets, mask=mask, other=0.0)

        dq_inter_offsets = block_start_d + tl.arange(0, BLOCK_D)
        dq_inter_offsets = dq_inter_offsets + (block_start_t + i) * stride_dq_inter_t
        dq_inter = tl.load(dq_inter_ptr + dq_inter_offsets, mask=mask, other=0.0)

        dk_inner_offsets = block_start_d + tl.arange(0, BLOCK_D)
        dk_inner_offsets = dk_inner_offsets + (block_start_t + i) * stride_dk_inner_t
        dk_inner = tl.load(dk_inner_ptr + dk_inner_offsets, mask=mask, other=0.0)

        dk_inter_offsets = block_start_d + tl.arange(0, BLOCK_D)
        dk_inter_offsets = dk_inter_offsets + (block_start_t + i) * stride_dk_inter_t
        dk_inter = tl.load(dk_inter_ptr + dk_inter_offsets, mask=mask, other=0.0)

        dq = dq_inner + dq_inter
        dk = dk_inner + dk_inter

        dg = dq * k + q * dk
        dg = dg * decay_factor

        dg_offsets = block_start_d + tl.arange(0, BLOCK_D)
        dg_offsets = dg_offsets + (block_start_t + i) * stride_dg_t
        tl.atomic_add(dg_ptr + dg_offsets, dg, mask=mask)

def launch_fwd_decay_cumsum(g, g_o, decay, BT, DK, T, BLOCK_D=128, BLOCK_T=128):
    grid = (DK // BLOCK_D, T // BLOCK_T, g.shape[0])
    strides = g.strides
    strides_o = g_o.strides
    fwd_decay_cumsum[grid](
        g, g_o, decay, BT, DK, T,
        strides[1], strides[2], strides_o[1], strides_o[2],
        BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T
    )
