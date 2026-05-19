import torch
import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    q,  # [B, H, T, K]
    k,  # [B, H, T, K]
    v,  # [B, H, T, V]
    o,  # [B, H, T, V]
    z,  # [B, H, T] or None when not using normalization
    s_q_h,  # q.stride(1)
    s_q_t,  # q.stride(2)
    s_q_k,  # q.stride(3)
    s_v_h,  # v.stride(1)
    s_v_t,  # v.stride(2)
    s_v_k,  # v.stride(3)
    scale,
    use_scale: tl.constexpr,
    use_normalize: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    # program_ids
    pid_kv = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_bh = tl.program_id(2)
    # compute block indices
    NK = tl.cdiv(K, BK)
    NV = tl.cdiv(V, BV)
    ih = pid_bh % H
    ib = pid_bh // H
    ik = pid_kv // NV
    iv = pid_kv % NV
    # block start
    seq_start = pid_seq * BTL
    # pointer for q [B, H, T, K]
    q_block_ptr = tl.make_block_ptr(
        base=q,
        shape=(B, H, T, K),
        strides=(q.stride(0), s_q_h, s_q_t, s_q_k),
        offsets=(ib, ih, seq_start, ik * BK),
        block_shape=(BTL, BK),
        order=(2, 3)
    )
    # pointer for k [B, H, T, K]
    k_block_ptr = tl.make_block_ptr(
        base=k,
        shape=(B, H, T, K),
        strides=(k.stride(0), k.stride(1), k.stride(2), k.stride(3)),
        offsets=(ib, ih, 0, ik * BK),
        block_shape=(BTS, BK),
        order=(0, 1)
    )
    # pointer for v [B, H, T, V]
    v_block_ptr = tl.make_block_ptr(
        base=v,
        shape=(B, H, T, V),
        strides=(v.stride(0), s_v_h, s_v_t, s_v_k),
        offsets=(ib, ih, 0, iv * BV),
        block_shape=(BTS, BV),
        order=(0, 1)
    )
    # pointer for output [B, H, T, V]
    o_block_ptr = tl.make_block_ptr(
        base=o,
        shape=(B, H, T, V),
        strides=(o.stride(0), o.stride(1), o.stride(2), o.stride(3)),
        offsets=(ib, ih, seq_start, iv * BV),
        block_shape=(BTL, BV),
        order=(0, 1)
    )
    # optional normalization factor z [B, H, T]
    z_block_ptr = tl.make_block_ptr(
        base=z,
        shape=(B, H, T),
        strides=(z.stride(0), z.stride(1), z.stride(2)) if use_normalize else (1, 1, 1),
        offsets=(ib, ih, seq_start),
        block_shape=(BTL,),
        order=(0,)
    ) if use_normalize else None

    # load local Q
    q_tile = tl.load(q_block_ptr, boundary_check=(0, 1))
    if use_scale:
        q_tile *= scale

    # accumulators for output
    out_accum = tl.zeros((BTL, BV), dtype=tl.float32)
    # accumulators for normalization
    norm_accum = tl.zeros((BTL,), dtype=tl.float32) if use_normalize else None

    # process k, v blocks
    # we iterate over the entire T dimension in steps of BTS
    # each step: gather partial attn scores, partial outputs
    # tileQ shape [BTL, BK], tileK shape [BTS, BK], tileV shape [BTS, BV]
    # partial attn = tileQ x tileK^T
    # then partial out = partial attn x tileV
    # accumulate in out_accum
    for ts in range(0, T, BTS):
        k_tile = tl.load(k_block_ptr, boundary_check=(0, 1))
        v_tile = tl.load(v_block_ptr, boundary_check=(0, 1))
        # [BTL, BTS]
        attn_scores = tl.dot(q_tile, tl.trans(k_tile))
        # optionally compute normalization factor
        if use_normalize:
            partial_sum = tl.sum(attn_scores, axis=1)
            norm_accum += partial_sum
        # [BTL, BV]
        out_partial = tl.dot(attn_scores.to(tl.float32), v_tile)
        out_accum += out_partial
        # advance pointers
        k_block_ptr = tl.advance(k_block_ptr, (BTS, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BTS, 0))

    # if normalization is used, store norm and normalize the output
    if use_normalize:
        tl.store(z_block_ptr, norm_accum, boundary_check=(0,))
        # read it back to normalize (avoid parallel read in loop)
        norm_val = tl.load(z_block_ptr, boundary_check=(0,))
        # guard against division by zero
        norm_val = tl.where(norm_val > 0, norm_val, 1.0)
        out_accum /= norm_val[:, None]

    # store the output
    tl.store(o_block_ptr, out_accum.to(o_block_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _parallel_rebased_bwd_dq(
    q, k, v,
    o, z,
    do, dq,
    q_strides,
    k_strides,
    v_strides,
    o_strides,
    z_strides,
    scale,
    use_scale: tl.constexpr,
    use_normalize: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    pid_kv = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_bh = tl.program_id(2)
    NK = tl.cdiv(K, BK)
    NV = tl.cdiv(V, BV)

    ih = pid_bh % H
    ib = pid_bh // H
    ik = pid_kv // NV
    iv = pid_kv % NV
    seq_start = pid_seq * BTL

    # block ptrs for do, q, k, v
    do_ptr = tl.make_block_ptr(
        base=do,
        shape=(B, H, T, V),
        strides=o_strides,
        offsets=(ib, ih, seq_start, iv * BV),
        block_shape=(BTL, BV),
        order=(0, 1)
    )
    q_ptr = tl.make_block_ptr(
        base=q,
        shape=(B, H, T, K),
        strides=q_strides,
        offsets=(ib, ih, seq_start, ik * BK),
        block_shape=(BTL, BK),
        order=(0, 1)
    )
    k_ptr = tl.make_block_ptr(
        base=k,
        shape=(B, H, T, K),
        strides=k_strides,
        offsets=(ib, ih, 0, ik * BK),
        block_shape=(BTS, BK),
        order=(0, 1)
    )
    v_ptr = tl.make_block_ptr(
        base=v,
        shape=(B, H, T, V),
        strides=v_strides,
        offsets=(ib, ih, 0, iv * BV),
        block_shape=(BTS, BV),
        order=(0, 1)
    )
    dq_ptr = tl.make_block_ptr(
        base=dq,
        shape=(B, H, T, K),
        strides=q_strides,
        offsets=(ib, ih, seq_start, ik * BK),
        block_shape=(BTL, BK),
        order=(0, 1)
    )

    z_ptr = tl.make_block_ptr(
        base=z,
        shape=(B, H, T),
        strides=z_strides if use_normalize else (1, 1, 1),
        offsets=(ib, ih, seq_start),
        block_shape=(BTL,),
        order=(0,)
    ) if use_normalize else None

    do_tile = tl.load(do_ptr, boundary_check=(0, 1))
    dq_accum = tl.zeros((BTL, BK), dtype=tl.float32)
    # possibly load normalization if used
    norm_tile = tl.load(z_ptr, boundary_check=(0,)) if use_normalize else None
    if use_normalize:
        norm_tile = tl.where(norm_tile > 0, norm_tile, 1.0)

    # accumulate gradient for Q
    for ts in range(0, T, BTS):
        k_tile = tl.load(k_ptr, boundary_check=(0, 1))       # [BTS, BK]
        v_tile = tl.load(v_ptr, boundary_check=(0, 1))       # [BTS, BV]
        # partial attention = dot(q_tile, k_tile.T)
        # backward w.r.t q => dot(do_tile, v_tile.T) * ...
        attn_scores
