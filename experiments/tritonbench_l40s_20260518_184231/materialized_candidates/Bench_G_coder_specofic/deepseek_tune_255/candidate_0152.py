import torch
import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    k, v, h, initial_state,
    b_h,
    s_qk_h, s_qk_t,
    s_vo_h, s_vo_t,
    H, T, scale,
    d_b_h, d_b_t,
    NT, B, K, V,
    initial_state_flag: tl.constexpr,
    final_state_flag: tl.constexpr
):
    i_k, i_v, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_h = tl.arange(0, K) % K
    mask_b = (i_b < B)
    mask_k = (i_k < K)
    mask_v = (i_v < V)
    mask_h = (o_h < H)

    p_q = tl.make_block_ptr(
        base=k + i_b * s_qk_h,
        shape=(K, T),
        strides=(1 * s_qk_h, 1 * s_qk_t),
        offsets=(0, 0),
        block_shape=(K, 1),
        order=(0, 1)
    )
    p_k = tl.make_block_ptr(
        base=k + i_b * s_qk_h,
        shape=(K, T),
        strides=(1 * s_qk_h, 1 * s_qk_t),
        offsets=(0, 0),
        block_shape=(1, K),
        order=(1, 0)
    )
    p_v = tl.make_block_ptr(
        base=v + i_b * s_vo_h,
        shape=(T, V),
        strides=(1 * s_vo_t, 1 * s_vo_h),
        offsets=(0, 0),
        block_shape=(1, V),
        order=(0, 1)
    )
    p_h = tl.make_block_ptr(
        base=h + i_b * K * T,
        shape=(K, T),
        strides=(1 * T, 1),
        offsets=(0, 0),
        block_shape=(K, 1),
        order=(0, 1)
    )
    p_b_h = tl.make_block_ptr(
        base=b_h + i_b * K * T,
        shape=(K, T),
        strides=(1 * T, 1),
        offsets=(0, 0),
        block_shape=(K, 1),
        order=(0, 1)
    )
    p_d_b_h = tl.make_block_ptr(
        base=d_b_h + i_b * K * T,
        shape=(K, T),
        strides=(1 * T, 1),
        offsets=(0, 0),
        block_shape=(K, 1),
        order=(0, 1)
    )

    mask_time = (0 < T)
    if initial_state_flag:
        p_init_h = tl.make_block_ptr(
            base=initial_state + i_b * K * T,
            shape=(K, T),
            strides=(1 * T, 1),
            offsets=(0, 0),
            block_shape=(K, 1),
            order=(0, 1)
        )
        b_h = tl.load(p_init_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(NT):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_k = tl.where(mask_k[None, :] & mask_time[None, None], b_k, 0)
        b_v = tl.where(mask_v[None, :] & mask_time[None, None], b_v, 0)
        d_b = tl.load(p_d_b_h, boundary_check=(0, 1))
        d_b = tl.where(mask_k[None, :] & mask_time[None, None], d_b, 0)
        b_k = b_k * scale
        b_h = tl.dot(b_k, b_h, allow_tf32=False) * d_b[None, :]
        b_o = tl.dot(b_k.to(tl.float16), b_v.to(tl.float16), allow_tf32=False)
        b_h = b_h.to(p_b_h.dtype.element_ty) + b_o.to(p_b_h.dtype.element_ty)
        tl.store(p_b_h, b_h, boundary_check=(0, 1))
        p_k = tl.advance(p_k, (0, 1))
        p_v = tl.advance(p_v, (1, 0))
        p_h = tl.advance(p_h, (0, 1))
        p_b_h = tl.advance(p_b_h, (0, 1))
        p_d_b_h = tl.advance(p_d_b_h, (0, 1))
        if i < NT - 1:
            tl.store(p_h, b_h, boundary_check=(0, 1))
            p_k = tl.advance(p_k, (-K, 1))
            p_v = tl.advance(p_v, (1, -V))
            p_h = tl.advance(p_h, (0, 1))
            p_b_h = tl.advance(p_b_h, (0, 1))
            p_d_b_h = tl.advance(p_d_b_h, (0, 1))

    if final_state_flag:
        p_final_h = tl.make_block_ptr(
            base=h + i_b * K * T + (K - H) * T,
            shape=(H, T),
            strides=(1 * T, 1),
            offsets=(0, 0),
            block_shape=(1, T),
            order=(0, 1)
        )
        tl.store(p_final_h, b_h.to(p_final_h.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_retention_fwd_kernel_o(
    q, k, v, h, o, s_qk_h, s_qk_t, s_vo_h, s_vo_t,
    H, T, scale, NT, B, K, V
):
    i_k, i_v, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_h = tl.arange(0, K) % K
    mask_b = (i_b < B)
    mask_k = (i_k < K)
    mask_v = (i_v < V)
    mask_h = (o_h < H)

    p_q = tl.make_block_ptr(
        base=q + i_b * s_qk_h,
        shape=(K, T),
        strides=(1 * s_qk_h, 1 * s_qk_t),
        offsets=(0, 0),
        block_shape=(K, 1),
