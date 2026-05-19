import triton
import triton.language as tl


# Forward kernel
@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, init_state_ptr, out_ptr, final_state_ptr,
    stride_qz, stride_qh, stride_qt, stride_qv,
    stride_kz, stride_kh, stride_kt, stride_kv,
    stride_vz, stride_vh, stride_vt, stride_vv,
    stride_oz, stride_oh, stride_ot, stride_ov,
    stride_sz, stride_sh, stride_st, stride_sv,
    T, N, H, D, scale,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr
):
    # Program IDs
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Offsets
    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)

    # Pointer to the chunk of data for this batch/head
    q_base_ptr = q_ptr + batch_id * stride_qz + head_id * stride_qh
    k_base_ptr = k_ptr + batch_id * stride_kz + head_id * stride_kh
    v_base_ptr = v_ptr + batch_id * stride_vz + head_id * stride_vh
    o_base_ptr = out_ptr + batch_id * stride_oz + head_id * stride_oh
    state_base_ptr = None
    if USE_INITIAL_STATE:
        state_base_ptr = init_state_ptr + batch_id * stride_sz + head_id * stride_sh
    final_state_base_ptr = None
    if STORE_FINAL_STATE:
        final_state_base_ptr = final_state_ptr + batch_id * stride_sz + head_id * stride_sh

    # Initialize hidden state h
    h_k = tl.zeros([BK], tl.float32)
    h_v = tl.zeros([BV], tl.float32)
    if USE_INITIAL_STATE:
        h_k = tl.load(state_base_ptr + offs_k * stride_st, mask=offs_k < D, other=0.)
        h_v = tl.load(state_base_ptr + offs_v * stride_st, mask=offs_v < D, other=0.)

    # Process time dimension
    for t in range(T):
        q_t = tl.load(q_base_ptr + t * stride_qt)
        k_t_k = tl.load(k_base_ptr + t * stride_kt + offs_k * stride_kv, mask=offs_k < D, other=0.)
        v_t_v = tl.load(v_base_ptr + t * stride_vt + offs_v * stride_vv, mask=offs_v < D, other=0.)

        # scaled query
        q_t = q_t * scale

        # Decay factor
        alpha = tl.exp(-q_t)
        # Update hidden state
        h_k = alpha * h_k + (1 - alpha) * k_t_k
        h_v = alpha * h_v + (1 - alpha) * v_t_v

        # Store output
        out_t_k = h_k
        out_t_v = h_v
        tl.store(o_base_ptr + t * stride_ot + offs_k * stride_ov, out_t_k, mask=offs_k < D)
        tl.store(o_base_ptr + t * stride_ot + offs_v * stride_ov, out_t_v, mask=offs_v < D)

    # Store final state if needed
    if STORE_FINAL_STATE:
        tl.store(final_state_base_ptr + offs_k * stride_st, h_k, mask=offs_k < D)
        tl.store(final_state_base_ptr + offs_v * stride_st, h_v, mask=offs_v < D)


# Backward kernel
@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q_ptr, k_ptr, v_ptr, dq_ptr, dk_ptr, dv_ptr,
    do_ptr, init_state_ptr, final_state_ptr,
    stride_qz, stride_qh, stride_qt, stride_qv,
    stride_kz, stride_kh, stride_kt, stride_kv,
    stride_vz, stride_vh, stride_vt, stride_vv,
    stride_dqz, stride_dqh, stride_dqt, stride_dqv,
    stride_dkz, stride_dkh, stride_dkt, stride_dkv,
    stride_dvz, stride_dvh, stride_dvt, stride_dvv,
    stride_doz, stride_doh, stride_dot, stride_dov,
    stride_sz, stride_sh, stride_st, stride_sv,
    T, N, H, D, scale,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_k = tl.arange(0, BK)
    offs_v = tl.arange(0, BV)

    q_base_ptr = q_ptr + batch_id * stride_qz + head_id * stride_qh
    k_base_ptr = k_ptr + batch_id * stride_kz + head_id * stride_kh
    v_base_ptr = v_ptr + batch_id * stride_vz + head_id * stride_vh
    dq_base_ptr = dq_ptr + batch_id * stride_dqz + head_id * stride_dqh
    dk_base_ptr = dk_ptr + batch_id * stride_dkz + head_id * stride_dkh
    dv_base_ptr = dv_ptr + batch_id * stride_dvz + head_id * stride_dvh
    do_base_ptr = do_ptr + batch_id * stride_doz + head_id * stride_doh

    state_base_ptr = None
    if USE_INITIAL_STATE:
        state_base_ptr = init_state_ptr + batch_id * stride_sz + head_id * stride_sh
    final_state_base_ptr = None
    if STORE_FINAL_STATE:
        final_state_base_ptr = final_state_ptr + batch_id * stride_sz + head_id * stride_sh

    # Reload final hidden state
    h_k = tl.zeros([BK], tl.float32)
    h_v = tl.zeros([BV], tl.float32)
    if STORE_FINAL_STATE:
        h_k = tl.load(final_state_base_ptr + offs_k * stride_st, mask=offs_k < D, other=0.)
        h_v = tl.load(final_state_base_ptr + offs_v * stride_st, mask=offs_v < D, other=0.)
    else:
        # If no final state stored, recompute forward pass or handle as zero
        # (this is an implementation detail requiring forward recomputation if needed)
        pass

    dh_k = tl.zeros([BK], tl.float32)
    dh_v = tl.zeros([BV], tl.float32)

    for t in range(T - 1, -1, -1):
        do_t_k = tl.load(do_base_ptr + t * stride_dot + offs_k * stride_dov, mask=offs_k < D, other=0.)
        do_t_v = tl.load(do_base_ptr + t * stride_dot + offs_v * stride_dov, mask=offs_v < D, other=0.)

        # Grad hidden
        dh_k += do_t_k
        dh_v += do_t_v

        # Load forward pass values
        q_t = tl.load(q_base_ptr + t * stride_qt)
        q_t_scaled = q_t * scale
        k_t_k = tl.load(k_base_ptr + t * stride_kt + offs_k * stride_kv, mask=offs_k < D, other=0.)
        v_t_v = tl.load(v_base_ptr + t * stride_vt + offs_v * stride_vv, mask=offs_v < D, other=0.)

        alpha = tl.exp(-q_t_scaled)
        # (1 - alpha) * x -> expand with chain rule
        d_alpha = (-tl.exp(-q_t_scaled)) * (scale * dq_t := 0.)

        # Partial derivatives
        # dh_k = alpha * dh_k + ...
        # Derivative w.r.t. k(t) is (1 - alpha)*dh_k
        dk = (1 - alpha) * dh_k
        dv = (1 - alpha) * dh_v

        # Derivative w.r.t. alpha
        d_alpha_k = (h_k - k_t_k) * dh_k
        d_alpha_v = (h_v - v_t_v) * dh_v
        d_alpha_total = d_alpha_k + d_alpha_v

        # d_alpha / d_q
        dq = d_alpha_total * d_alpha

        # Write back
        tl.atomic_add(dq_base_ptr + t * stride_dqt, dq)
        tl.atomic_add(dk_base_ptr + t * stride_dkt + offs_k * stride_dkv, dk, mask=offs_k < D)
        tl.atomic_add(dv_base_ptr + t * stride_dvt + offs_v * stride_dvv, dv, mask=offs_v < D)

        # Update hidden grad
        dh_k = alpha * dh_k
        dh_v = alpha * dh_v

        # Recompute old h for next iteration
        # Substituting simpler version here as a placeholder
        old_h_k = (h_k - (1 - alpha) * k_t_k) / alpha
        old_h_v = (h_v - (1 - alpha) * v_t_v) / alpha
        h_k = old_h_k
        h_v = old_h_v

    # Handle initial state gradient if we used it
    if USE_INITIAL_STATE:
        tl.atomic_add(dq_base_ptr, 0)  # placeholder if needed for init state


def fused_recurrent_retention(q, k, v, initial_state=None, scale=1.0, store_final_state=False):
    """
    Fused recurrent retention wrapper.
    """
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Tensors must be contiguous."
    B, H, T, D = q.shape

    USE_INITIAL_STATE = initial_state is not None
    STORE_FINAL_STATE = store_final_state

    # Create output tensor
    import torch
    o = torch.empty_like(q)
    final_state = None
    if STORE_FINAL_STATE:
        final_state = torch.empty((B, H, D), device=q.device, dtype=q.dtype)

    # Grid
    grid = (B, H)

    BK = 32
    BV = 32

    # Forward launch
    def fwd_kernel_call():
        triton.run(
            fused_recurrent_retention_fwd_kernel,
            grid=grid,
            num_warps=4,
            num_stages=2,
            args=[
                q, k, v,
                initial_state if USE_INITIAL_STATE else 0,
                o,
                final_state if STORE_FINAL_STATE else 0,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                0, 0, 0, 0,  # placeholder for state strides
                T, B, H, D, scale,
                USE_INITIAL_STATE, STORE_FINAL_STATE,
                BK, BV
            ]
        )

    fwd_kernel_call()
    return (o, final_state) if STORE_FINAL_STATE else (o,)


def fused_recurrent_retention_backward(q, k, v, o, do, initial_state=None, final_state=None, scale=1.0):
    """
    Backward pass wrapper.
    """
    import torch
    B, H, T, D = q.shape

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    USE_INITIAL_STATE = initial_state is not None
    STORE_FINAL_STATE = final_state is not None

    grid = (B, H)

    BK = 32
    BV = 32

    def bwd_kernel_call():
        triton.run(
            fused_recurrent_retention_bwd_kernel,
            grid=grid,
            num_warps=4,
            num_stages=2,
            args=[
                q, k, v,
                dq, dk, dv,
                do,
                initial_state if USE_INITIAL_STATE else 0,
                final_state if STORE_FINAL_STATE else 0,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                0, 0, 0, 0,  # placeholder for state strides
                T, B, H, D, scale,
                USE_INITIAL_STATE, STORE_FINAL_STATE,
                BK, BV
            ]
        )

    bwd_kernel_call()
    return dq, dk, dv
