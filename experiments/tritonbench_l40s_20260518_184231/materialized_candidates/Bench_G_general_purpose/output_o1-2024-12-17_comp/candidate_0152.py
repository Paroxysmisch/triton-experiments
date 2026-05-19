import math
import torch
import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    K_ptr, V_ptr,
    H_ptr,  # b_h
    INITIAL_STATE_ptr,
    FINAL_STATE_ptr,
    BATCH, HEAD, NT, SIZE,
    stride_kb, stride_kh, stride_kt,
    stride_vb, stride_vh, stride_vt,
    stride_hb, stride_hh, stride_ht,
    stride_isb, stride_ish, stride_ist,
    stride_fsb, stride_fsh, stride_fst,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    off_t = tl.arange(0, BLOCK_SIZE)
    off_base = batch_idx * stride_hb + head_idx * stride_hh
    H_offset = off_base
    if USE_INITIAL_STATE != 0:
        init_offset = batch_idx * stride_isb + head_idx * stride_ish
        # load initial state
        state = tl.load(INITIAL_STATE_ptr + init_offset + off_t * stride_ist, mask=off_t < SIZE, other=0.0)
    else:
        state = 0.0

    for chunk_start in range(0, NT, BLOCK_SIZE):
        t_mask = off_t + chunk_start
        # load data
        K_offset = batch_idx * stride_kb + head_idx * stride_kh + t_mask * stride_kt
        V_offset = batch_idx * stride_vb + head_idx * stride_vh + t_mask * stride_vt
        k = tl.load(K_ptr + K_offset, mask=t_mask < NT, other=0.0)
        v = tl.load(V_ptr + V_offset, mask=t_mask < NT, other=0.0)
        # decay factors (dummy example, replace as needed)
        d_b = 1.0 / (1.0 + k * 0.1)
        d_i = 1.0 / (1.0 + v * 0.1)
        # update state
        state = state * d_b + k * v * d_i
        # store result in H
        H_ptr_offset = H_offset + (chunk_start + off_t) * stride_ht
        tl.store(H_ptr + H_ptr_offset, state, mask=t_mask < NT)

    if STORE_FINAL_STATE != 0:
        final_offset = batch_idx * stride_fsb + head_idx * stride_fsh
        tl.store(FINAL_STATE_ptr + final_offset + off_t * stride_fst, state, mask=off_t < SIZE)

@triton.jit
def chunk_retention_fwd_kernel_o(
    Q_ptr, K_ptr, V_ptr,
    H_ptr,
    O_ptr,
    BATCH, HEAD, NT, SIZE,
    stride_qb, stride_qh, stride_qt,
    stride_kb, stride_kh, stride_kt,
    stride_vb, stride_vh, stride_vt,
    stride_hb, stride_hh, stride_ht,
    stride_ob, stride_oh, stride_ot,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    off_t = tl.arange(0, BLOCK_SIZE)
    t_mask = off_t

    q_offset = batch_idx * stride_qb + head_idx * stride_qh + off_t * stride_qt
    Qv = tl.load(Q_ptr + q_offset, mask=t_mask < NT, other=0.0)

    # Aggregation buffers
    b_o = 0.0
    b_s = 0.0

    for chunk_start in range(0, NT, BLOCK_SIZE):
        K_offset = batch_idx * stride_kb + head_idx * stride_kh + (chunk_start + off_t) * stride_kt
        V_offset = batch_idx * stride_vb + head_idx * stride_vh + (chunk_start + off_t) * stride_vt
        H_offset = batch_idx * stride_hb + head_idx * stride_hh + (chunk_start + off_t) * stride_ht
        k = tl.load(K_ptr + K_offset, mask=(chunk_start + off_t) < NT, other=0.0)
        v = tl.load(V_ptr + V_offset, mask=(chunk_start + off_t) < NT, other=0.0)
        h = tl.load(H_ptr + H_offset, mask=(chunk_start + off_t) < NT, other=0.0)
        # decay factors (example placeholder)
        d_i = 1.0 / (1.0 + (k + v) * 0.1)
        contrib = Qv * k * v * d_i + h
        b_o += contrib
        b_s += 1.0

    # finalize output
    out = b_o / b_s
    o_offset = batch_idx * stride_ob + head_idx * stride_oh + off_t * stride_ot
    tl.store(O_ptr + o_offset, out, mask=t_mask < NT)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    DH_ptr,
    DHO_ptr,  # partial updates to dh from backward pass
    BATCH, HEAD, NT, SIZE,
    stride_dhb, stride_dhh, stride_dht,
    stride_dhob, stride_dhoh, stride_dhot,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    off_t = tl.arange(0, BLOCK_SIZE)
    # We iterate backwards in time
    state = 0.0
    for chunk_start in range(NT - BLOCK_SIZE, -1, -BLOCK_SIZE):
        t_mask = off_t + chunk_start
        dho_offset = batch_idx * stride_dhob + head_idx * stride_dhoh + t_mask * stride_dhot
        dho_val = tl.load(DHO_ptr + dho_offset, mask=t_mask >= 0, other=0.0)
        # accumulate
        state = state + dho_val
        dh_offset = batch_idx * stride_dhb + head_idx * stride_dhh + t_mask * stride_dht
        tl.store(DH_ptr + dh_offset, state, mask=t_mask >= 0)

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    Q_ptr, K_ptr, V_ptr,
    DQ_ptr, DK_ptr, DV_ptr,
    DH_ptr,
    BATCH, HEAD, NT, SIZE,
    stride_qb, stride_qh, stride_qt,
    stride_kb, stride_kh, stride_kt,
    stride_vb, stride_vh, stride_vt,
    stride_dqb, stride_dqh, stride_dqt,
    stride_dkb, stride_dkh, stride_dkt,
    stride_dvb, stride_dvh, stride_dvt,
    stride_dhb, stride_dhh, stride_dht,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    off_t = tl.arange(0, BLOCK_SIZE)
    t_mask = off_t

    q_offset = batch_idx * stride_qb + head_idx * stride_qh + off_t * stride_qt
    k_offset = batch_idx * stride_kb + head_idx * stride_kh + off_t * stride_kt
    v_offset = batch_idx * stride_vb + head_idx * stride_vh + off_t * stride_vt
    dh_offset = batch_idx * stride_dhb + head_idx * stride_dhh + off_t * stride_dht

    q = tl.load(Q_ptr + q_offset, mask=t_mask < NT, other=0.0)
    k = tl.load(K_ptr + k_offset, mask=t_mask < NT, other=0.0)
    v = tl.load(V_ptr + v_offset, mask=t_mask < NT, other=0.0)
    dh = tl.load(DH_ptr + dh_offset, mask=t_mask < NT, other=0.0)

    # Example gradient logic
    d_i = 1.0 / (1.0 + (k + v) * 0.1)
    grad_contrib = dh * d_i
    # partial derivatives
    dq = grad_contrib * k * v
    dk = grad_contrib * q * v
    dv = grad_contrib * q * k

    dq_offset = batch_idx * stride_dqb + head_idx * stride_dqh + off_t * stride_dqt
    dk_offset = batch_idx * stride_dkb + head_idx * stride_dkh + off_t * stride_dkt
    dv_offset = batch_idx * stride_dvb + head_idx * stride_dvh + off_t * stride_dvt

    tl.store(DQ_ptr + dq_offset, dq, mask=t_mask < NT)
    tl.store(DK_ptr + dk_offset, dk, mask=t_mask < NT)
    tl.store(DV_ptr + dv_offset, dv, mask=t_mask < NT)

class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, use_initial_state=False, initial_state=None, store_final_state=False):
        B, H, T, D = q.shape
        # Flatten for kernel usage
        q_ = q.contiguous()
        k_ = k.contiguous()
        v_ = v.contiguous()

        # Prepare memory for states if needed
        if use_initial_state:
            initial_state_ = initial_state.contiguous()
        else:
            initial_state_ = torch.zeros_like(q[:, :, 0, :])

        h_ = torch.zeros_like(k_)
        o_ = torch.zeros_like(q_)

        final_state_ = None
        if store_final_state:
            final_state_ = torch.zeros_like(initial_state_)

        grid = (B, H)
        block_size = D

        # Launch fwd kernel h
        chunk_retention_fwd_kernel_h[grid](
            k_.data_ptr(), v_.data_ptr(),
            h_.data_ptr(),
            initial_state_.data_ptr() if use_initial_state else 0,
            final_state_.data_ptr() if store_final_state else 0,
            B, H, T, D,
            k_.stride(0), k_.stride(1), k_.stride(2),
            v_.stride(0), v_.stride(1), v_.stride(2),
            h_.stride(0), h_.stride(1), h_.stride(2),
            initial_state_.stride(0), initial_state_.stride(1), initial_state_.stride(2),
            final_state_.stride(0) if store_final_state else 0,
            final_state_.stride(1) if store_final_state else 0,
            final_state_.stride(2) if store_final_state else 0,
            use_initial_state, store_final_state, BLOCK_SIZE=block_size
        )

        # Launch fwd kernel o
        chunk_retention_fwd_kernel_o[grid](
            q_.data_ptr(), k_.data_ptr(), v_.data_ptr(),
            h_.data_ptr(),
            o_.data_ptr(),
            B, H, T, D,
            q_.stride(0), q_.stride(1), q_.stride(2),
            k_.stride(0), k_.stride(1), k_.stride(2),
            v_.stride(0), v_.stride(1), v_.stride(2),
            h_.stride(0), h_.stride(1), h_.stride(2),
            o_.stride(0), o_.stride(1), o_.stride(2),
            BLOCK_SIZE=block_size
        )

        ctx.save_for_backward(q_, k_, v_, h_)
        ctx.use_initial_state = use_initial_state
        ctx.store_final_state = store_final_state
        ctx.initial_state_ = initial_state_
        ctx.final_state_ = final_state_

        return (o_, final_state_) if store_final_state else (o_, None)

    @staticmethod
    def backward(ctx, grad_o, grad_final_state=None):
        q_, k_, v_, h_ = ctx.saved_tensors
        B, H, T, D = q_.shape

        dq = torch.zeros_like(q_)
        dk = torch.zeros_like(k_)
        dv = torch.zeros_like(v_)
        dh = torch.zeros_like(h_)

        # Backward pass kernels
        grid = (B, H)
        block_size = D

        # 1) compute dh
        chunk_retention_bwd_kernel_dh[grid](
            dh.data_ptr(),
            grad_o.data_ptr(),  # example usage for dho
            B, H, T, D,
            dh.stride(0), dh.stride(1), dh.stride(2),
            grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
            BLOCK_SIZE=block_size
        )

        # 2) compute dq, dk, dv
        chunk_retention_bwd_kernel_dqkv[grid](
            q_.data_ptr(), k_.data_ptr(), v_.data_ptr(),
            dq.data_ptr(), dk.data_ptr(), dv.data_ptr(),
            dh.data_ptr(),
            B, H, T, D,
            q_.stride(0), q_.stride(1), q_.stride(2),
            k_.stride(0), k_.stride(1), k_.stride(2),
            v_.stride(0), v_.stride(1), v_.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            dh.stride(0), dh.stride(1), dh.stride(2),
            BLOCK_SIZE=block_size
        )

        return dq, dk, dv, None, None, None

def chunk_retention(q, k, v, use_initial_state=False, initial_state=None, store_final_state=False):
    return ChunkRetentionFunction.apply(q, k, v, use_initial_state, initial_state, store_final_state)
