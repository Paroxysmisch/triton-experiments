import torch
import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    Q_PTR, K_PTR, V_PTR, H_PTR,
    T, K, V,
    stride_qz, stride_qh, stride_qt, stride_qk,
    stride_kz, stride_kh, stride_kt, stride_kk,
    stride_vz, stride_vh, stride_vt, stride_vk,
    stride_hz, stride_hh, stride_ht, stride_hk,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
    start_idx_t: tl.constexpr
):
    # Program IDs
    pid_t = tl.program_id(0)
    pid_z = tl.program_id(1)

    # Offsets
    t_offset = pid_t * BLOCK_T + start_idx_t
    # Boundary check
    mask_t = t_offset + tl.arange(0, BLOCK_T)
    valid_t = mask_t < T

    # Pointers for Q
    q_ptrs = Q_PTR + pid_z*stride_qz + 0*stride_qh + mask_t*stride_qt

    # Initialize accumulators
    h_acc = tl.zeros([BLOCK_T, BLOCK_K], dtype=tl.float32)

    # Iterate over K dimension in blocks
    for kk in range(0, K, BLOCK_K):
        mask_k = kk + tl.arange(0, BLOCK_K)
        valid_k = mask_k < K

        # Block pointers
        k_ptrs = K_PTR + pid_z*stride_kz + 0*stride_kh + mask_k*stride_kk
        v_ptrs = V_PTR + pid_z*stride_vz + 0*stride_vh + mask_k*stride_vk

        # Load Q, K, V
        q_val = tl.load(q_ptrs + 0*stride_qk, mask=valid_t)
        k_val = tl.load(k_ptrs + 0*stride_kt, mask=valid_k)
        v_val = tl.load(v_ptrs + 0*stride_vt, mask=valid_k)

        # Broadcast for dot product
        q_val_2d = tl.broadcast_to(q_val[:, None], [BLOCK_T, BLOCK_K])
        k_val_2d = tl.broadcast_to(k_val[None, :], [BLOCK_T, BLOCK_K])
        dot_qk = q_val_2d * k_val_2d

        # Accumulate
        h_acc += dot_qk * v_val[None, :]

    # Write to H
    h_ptrs = H_PTR + pid_z*stride_hz + 0*stride_hh + mask_t*stride_ht
    tl.store(h_ptrs, h_acc, mask=valid_t)


@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    Q_PTR, K_PTR, V_PTR, H_PTR, O_PTR,
    T, K, V,
    stride_qz, stride_qh, stride_qt, stride_qk,
    stride_kz, stride_kh, stride_kt, stride_kk,
    stride_vz, stride_vh, stride_vt, stride_vk,
    stride_hz, stride_hh, stride_ht, stride_hk,
    stride_oz, stride_oh, stride_ot, stride_ok,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    start_idx_t: tl.constexpr
):
    pid_t = tl.program_id(0)
    pid_z = tl.program_id(1)

    t_offset = pid_t * BLOCK_T + start_idx_t
    mask_t = t_offset + tl.arange(0, BLOCK_T)
    valid_t = mask_t < T

    q_ptrs = Q_PTR + pid_z*stride_qz + 0*stride_qh + mask_t*stride_qt

    o_acc = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)

    for kk in range(0, K, BLOCK_K):
        mask_k = kk + tl.arange(0, BLOCK_K)
        valid_k = mask_k < K

        k_ptrs = K_PTR + pid_z*stride_kz + 0*stride_kh + mask_k*stride_kk
        v_ptrs = V_PTR + pid_z*stride_vz + 0*stride_vh + mask_k*stride_vk
        h_ptrs = H_PTR + pid_z*stride_hz + 0*stride_hh + mask_t*stride_ht

        q_val = tl.load(q_ptrs + 0*stride_qk, mask=valid_t)
        k_val = tl.load(k_ptrs + 0*stride_kt, mask=valid_k)
        v_val = tl.load(v_ptrs + 0*stride_vt, mask=valid_k)
        h_val = tl.load(h_ptrs, mask=valid_t)

        q_val_2d = tl.broadcast_to(q_val[:, None], [BLOCK_T, BLOCK_K])
        k_val_2d = tl.broadcast_to(k_val[None, :], [BLOCK_T, BLOCK_K])
        attn_score = q_val_2d * k_val_2d  # raw dot
        attn_score = tl.where(valid_k[None, :], attn_score, 0.)

        # Weighted sum
        weight = attn_score / (h_val + 1e-9)
        value = tl.broadcast_to(v_val[None, :], [BLOCK_T, BLOCK_K])
        part_o = weight * value
        o_acc += tl.sum(part_o, 1)[:, None]

    o_ptrs = O_PTR + pid_z*stride_oz + 0*stride_oh + mask_t*stride_ot
    tl.store(o_ptrs, o_acc, mask=valid_t)


@triton.jit
def chunk_linear_attn_bwd_kernel_dh(
    DO_PTR, Q_PTR, K_PTR, V_PTR, H_PTR, DH_PTR,
    T, K, V,
    stride_doz, stride_doh, stride_dot, stride_dok,
    stride_qz, stride_qh, stride_qt, stride_qk,
    stride_kz, stride_kh, stride_kt, stride_kk,
    stride_vz, stride_vh, stride_vt, stride_vk,
    stride_hz, stride_hh, stride_ht, stride_hk,
    stride_dhz, stride_dhh, stride_dht, stride_dhk,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
    start_idx_t: tl.constexpr
):
    pid_t = tl.program_id(0)
    pid_z = tl.program_id(1)

    t_offset = pid_t * BLOCK_T + start_idx_t
    mask_t = t_offset + tl.arange(0, BLOCK_T)
    valid_t = mask_t < T

    do_ptrs = DO_PTR + pid_z*stride_doz + 0*stride_doh + mask_t*stride_dot
    dh_acc = tl.zeros([BLOCK_T, BLOCK_K], dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        mask_k = kk + tl.arange(0, BLOCK_K)
        valid_k = mask_k < K
        do_val = tl.load(do_ptrs, mask=valid_t)
        dh_acc += do_val[:, None]  # simple placeholder

    dh_ptrs = DH_PTR + pid_z*stride_dhz + 0*stride_dhh + mask_t*stride_dht
    tl.store(dh_ptrs, dh_acc, mask=valid_t)


@triton.jit
def chunk_linear_attn_bwd_kernel_dqkv(
    DO_PTR, DH_PTR, Q_PTR, K_PTR, V_PTR,
    DQ_PTR, DK_PTR, DV_PTR,
    T, K, V,
    stride_doz, stride_doh, stride_dot, stride_dok,
    stride_dhz, stride_dhh, stride_dht, stride_dhk,
    stride_qz, stride_qh, stride_qt, stride_qk,
    stride_kz, stride_kh, stride_kt, stride_kk,
    stride_vz, stride_vh, stride_vt, stride_vk,
    stride_dqz, stride_dqh, stride_dqt, stride_dqk,
    stride_dkz, stride_dkh, stride_dkt, stride_dkk,
    stride_dvz, stride_dvh, stride_dvt, stride_dvk,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
    start_idx_t: tl.constexpr
):
    pid_t = tl.program_id(0)
    pid_z = tl.program_id(1)

    t_offset = pid_t * BLOCK_T + start_idx_t
    mask_t = t_offset + tl.arange(0, BLOCK_T)
    valid_t = mask_t < T

    do_ptrs = DO_PTR + pid_z*stride_doz + 0*stride_doh + mask_t*stride_dot
    dh_ptrs = DH_PTR + pid_z*stride_dhz + 0*stride_dhh + mask_t*stride_dht

    dq_acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        mask_k = kk + tl.arange(0, BLOCK_K)
        valid_k = mask_k < K
        do_val = tl.load(do_ptrs, mask=valid_t)
        dh_val = tl.load(dh_ptrs, mask=valid_t)
        dq_acc += do_val + tl.sum(dh_val, 1)

    dq_ptrs = DQ_PTR + pid_z*stride_dqz + 0*stride_dqh + mask_t*stride_dqt
    tl.store(dq_ptrs, dq_acc, mask=valid_t)


class ChunkLinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        B, T, C = q.shape
        _, _, _ = k.shape
        _, _, _ = v.shape

        BLOCK_T = 128
        BLOCK_K = 64
        BLOCK_V = 64

        h = torch.zeros_like(q, dtype=q.dtype)
        o = torch.zeros_like(q, dtype=q.dtype)

        grid_fwd_h = ( (T + BLOCK_T - 1) // BLOCK_T, B )
        grid_fwd_o = ( (T + BLOCK_T - 1) // BLOCK_T, B )

        chunk_linear_attn_fwd_kernel_h[grid_fwd_h](
            q, k, v, h,
            T, C, C,
            q.stride(0), q.stride(1), q.stride(2), 1,
            k.stride(0), k.stride(1), k.stride(2), 1,
            v.stride(0), v.stride(1), v.stride(2), 1,
            h.stride(0), h.stride(1), h.stride(2), 1,
            BLOCK_T, BLOCK_K,
            0
        )

        chunk_linear_attn_fwd_kernel_o[grid_fwd_o](
            q, k, v, h, o,
            T, C, C,
            q.stride(0), q.stride(1), q.stride(2), 1,
            k.stride(0), k.stride(1), k.stride(2), 1,
            v.stride(0), v.stride(1), v.stride(2), 1,
            h.stride(0), h.stride(1), h.stride(2), 1,
            o.stride(0), o.stride(1), o.stride(2), 1,
            BLOCK_T, BLOCK_K, BLOCK_V,
            0
        )

        ctx.save_for_backward(q, k, v, h)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, h = ctx.saved_tensors
        B, T, C = q.shape
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        dh = torch.zeros_like(q)
        BLOCK_T = 128
        BLOCK_K = 64

        grid_bwd_dh = ( (T + BLOCK_T - 1) // BLOCK_T, B )
        chunk_linear_attn_bwd_kernel_dh[grid_bwd_dh](
            do, q, k, v, h, dh,
            T, C, C,
            do.stride(0), do.stride(1), do.stride(2), 1,
            q.stride(0), q.stride(1), q.stride(2), 1,
            k.stride(0), k.stride(1), k.stride(2), 1,
            v.stride(0), v.stride(1), v.stride(2), 1,
            h.stride(0), h.stride(1), h.stride(2), 1,
            dh.stride(0), dh.stride(1), dh.stride(2), 1,
            BLOCK_T, BLOCK_K,
            0
        )

        grid_bwd_dqkv = ( (T + BLOCK_T - 1) // BLOCK_T, B )
        chunk_linear_attn_bwd_kernel_dqkv[grid_bwd_dqkv](
            do, dh, q, k, v, dq, dk, dv,
            T, C, C,
            do.stride(0), do.stride(1), do.stride(2), 1,
            dh.stride(0), dh.stride(1), dh.stride(2), 1,
            q.stride(0), q.stride(1), q.stride(2), 1,
            k.stride(0), k.stride(1), k.stride(2), 1,
            v.stride(0), v.stride(1), v.stride(2), 1,
            dq.stride(0), dq.stride(1), dq.stride(2), 1,
            dk.stride(0), dk.stride(1), dk.stride(2), 1,
            dv.stride(0), dv.stride(1), dv.stride(2), 1,
            BLOCK_T, BLOCK_K,
            0
        )

        return dq, dk, dv


def chunk_linear_attention(q, k, v):
    return ChunkLinearAttentionFunction.apply(q, k, v)
