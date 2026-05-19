import torch
import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, K]
    v,  # value [B, H, L, V]
    o,  # output [B, H, L, V]
    z,  # normalization factor [B, H, L]
    s_qk_h,  # stride size: L * K
    s_qk_t,  # stride size: K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L * V
    s_vo_t,  # stride size: V
    s_vo_d,  # stride size: 1
    scale,  # scaling factor
    use_scale: tl.constexpr,  # whether to use scaling
    use_normalize: tl.constexpr,  # whether to use normalization
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # number of heads
    T: tl.constexpr,  # sequence length
    K: tl.constexpr,  # feature dimension for q and k
    V: tl.constexpr,  # feature dimension for v
    BTL: tl.constexpr,  # block size for sequence dimension in q
    BTS: tl.constexpr,  # block size for sequence dimension in k/v
    BK: tl.constexpr,  # block size for feature dimension in q/k
    BV: tl.constexpr,  # block size for feature dimension in v
):
    # Get program ids
    i_bh, i_c, i_kv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    # Create block pointers
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, 0), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, 0), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, 0), (BTS, BV), (1, 0))

    # Load query block
    b_q = tl.load(p_q, boundary_check=(0, 1))
    if use_scale:
        b_q *= scale

    # Initialize output and normalization factor
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    # Loop over key-value blocks
    for i in range(0, T, BTS):
        # Load key and value blocks
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # Compute attention scores
        b_s = tl.dot(b_q, b_k, allow_tf32=False)

        # Apply normalization if needed
        if use_normalize:
            b_s_max = tl.max(b_s, axis=1)
            b_s = b_s - b_s_max[:, None]
            b_s_exp = tl.exp(b_s)
            b_z = b_z + tl.sum(b_s_exp, axis=1)
            b_s = b_s_exp / b_z[:, None]

        # Compute output block
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)

        # Advance pointers
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (BTS, 0))

    # Store results
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, 0), (BTL, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    if use_normalize:
        p_z = tl.make_block_ptr(z + i_bh * s_qk_h, (T,), (s_qk_t,), (i_c * BTL,), (BTL,), (1,))
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def parallel_rebased_bwd_kernel(
    q, k, v, do, dz, dq, dk, dv,
    s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    scale, use_scale: tl.constexpr, use_normalize: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    i_bh, i_c, i_kv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    # Create block pointers
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, 0), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, 0), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, 0), (BTS, BV), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, 0), (BTL, BV), (1, 0))

    # Load blocks
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    b_dk = tl.zeros([BTS, BK], dtype=tl.float32)
    b_dv = tl.zeros([BTS, BV], dtype=tl.float32)

    # Loop over key-value blocks
    for i in range(0, T, BTS):
        # Load key and value blocks
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # Compute gradients
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)

        b_dk += tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)
        b_dv += tl.dot(b_ds.to(b_v.dtype), b_v, allow_tf32=False)

        # Advance pointers
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (BTS, 0))

    # Store gradients
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, 0), (BTL, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, 0), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, 0), (BTL, BV), (1, 0))

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty
