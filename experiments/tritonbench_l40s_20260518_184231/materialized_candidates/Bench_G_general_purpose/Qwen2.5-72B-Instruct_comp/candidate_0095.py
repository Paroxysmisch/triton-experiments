import triton
import triton.language as tl

@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, initial_state_ptr, final_state_ptr,
    T, B, H, D, BK, BV, scale, USE_INITIAL_STATE, STORE_FINAL_STATE,
    BLOCK_T: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    t_offset = (pid % T) * BLOCK_T
    b_offset = (pid // T) % B * BLOCK_B
    h_offset = (pid // (T * B)) % H * BLOCK_H
    d_offset = (pid // (T * B * H)) * BLOCK_D

    q = tl.load(q_ptr + t_offset * B * H * D + b_offset * H * D + h_offset * D + d_offset, mask=t_offset + tl.arange(0, BLOCK_T) < T, other=0.0)
    k = tl.load(k_ptr + t_offset * B * H * D + b_offset * H * D + h_offset * D + d_offset, mask=t_offset + tl.arange(0, BLOCK_T) < T, other=0.0)
    v = tl.load(v_ptr + t_offset * B * H * D + b_offset * H * D + h_offset * D + d_offset, mask=t_offset + tl.arange(0, BLOCK_T) < T, other=0.0)

    h = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    if USE_INITIAL_STATE:
        initial_state = tl.load(initial_state_ptr + b_offset * H * D + h_offset * D + d_offset, mask=t_offset + tl.arange(0, BLOCK_T) < T, other=0.0)
        h = initial_state

    decay = tl.exp(-scale * (h_offset + tl.arange(0, BLOCK_H)))

    for t in range(T):
        q_t = q[t, :]
        k_t = k[t, :]
        v_t = v[t, :]
        s = tl.sum(q_t * k_t, axis=1) * scale
        h = h * decay + s * v_t
        tl.store(o_ptr + t * B * H * D + b_offset * H * D + h_offset * D + d_offset, h, mask=t + tl.arange(0, BLOCK_T) < T)

    if STORE_FINAL_STATE:
        tl.store(final_state_ptr + b_offset * H * D + h_offset * D + d_offset, h, mask=t_offset + tl.arange(0, BLOCK_T) < T)

@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
    T, B, H, D, BK, BV, scale,
    BLOCK_T: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    t_offset = (pid % T) * BLOCK_T
    b_offset = (pid // T) % B * BLOCK_B
    h_offset = (pid // (T * B)) % H * BLOCK_H
    d_offset = (pid // (T * B * H)) * BLOCK_D

    q = tl.load(q_ptr + t_offset * B * H * D + b_offset * H * D + h_offset * D + d_offset, mask=t_offset + tl.arange(0, BLOCK_T) < T, other=0.0)
    k = tl.load(k_ptr + t_offset * B * H * D + b_offset * H * D + h_offset * D + d_offset, mask=t_offset + tl.arange(0, BLOCK_T) < T, other=0.0)
    v = tl.load(v_ptr + t_offset * B * H * D + b_offset * H * D + h_offset * D + d_offset, mask=t_offset + tl.arange(0, BLOCK_T) < T, other=0.0)
    o = tl.load(o_ptr + t_offset * B * H * D + b_offset * H * D + h_offset * D + d_offset, mask=t_offset + tl.arange(0, BLOCK_T) < T, other=0.0)
    do = tl.load(do_ptr + t_offset * B * H * D + b_offset * H * D + h_offset * D + d_offset, mask=t_offset + tl.arange(0, BLOCK_T) < T, other=0.0)

    dh = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    decay = tl.exp(-scale * (h_offset + tl.arange(0, BLOCK_H)))

    for t in range(T-1, -1, -1):
        q_t = q[t, :]
        k_t = k[t, :]
        v_t = v[t, :]
        o_t = o[t, :]
        do_t = do[t, :]
        s = tl.sum(q_t * k_t, axis=1) * scale
        dh = dh * decay + do_t
        dv = s * dh
        ds = dv * v_t
        dk = ds * q_t * scale
        dq = ds * k_t * scale
        tl.store(dq_ptr + t * B * H * D + b_offset * H * D + h_offset * D + d_offset, dq, mask=t + tl.arange(0, BLOCK_T) < T)
        tl.store(dk_ptr + t * B * H * D + b_offset * H * D + h_offset * D + d_offset, dk, mask=t + tl.arange(0, BLOCK_T) < T)
        tl.store(dv_ptr + t * B * H * D + b_offset * H * D + h_offset * D + d_offset, dv, mask=t + tl.arange(0, BLOCK_T) < T)

import torch

def fused_recurrent_retention(q, k, v, initial_state=None, store_final_state=False):
    T, B, H, D = q.shape
    BK, BV = 32, 32  # Block sizes
    scale = 1.0 / (D ** 0.5)

    o = torch.empty_like(q)
    final_state = torch.empty((B, H, D), device=q.device) if store_final_state else None

    grid = (T * B * H * (D // BK),)
    fused_recurrent_retention_fwd_kernel[grid](
        q, k, v, o, initial_state, final_state,
        T, B, H, D, BK, BV, scale, initial_state is not None, store_final_state,
        BLOCK_T=1, BLOCK_B=B, BLOCK_H=H, BLOCK_D=BK
    )

    return o, final_state

def fused_recurrent_retention_backward(q, k, v, o, do, initial_state=None):
    T, B, H, D = q.shape
    BK, BV = 32, 32  # Block sizes
    scale = 1.0 / (D ** 0.5)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    grid = (T * B * H * (D // BK),)
    fused_recurrent_retention_bwd_kernel[grid](
        q, k, v, o, do, dq, dk, dv,
        T, B, H, D, BK, BV, scale,
        BLOCK_T=1, BLOCK_B=B, BLOCK_H=H, BLOCK_D=BK
    )

    return dq, dk, dv
