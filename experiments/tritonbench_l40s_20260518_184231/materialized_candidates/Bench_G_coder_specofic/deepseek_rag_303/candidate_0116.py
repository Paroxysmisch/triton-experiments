import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BT": 16, "BK": 16, "BV": 16}, num_warps=4, num_stages=2),
        triton.Config({"BT": 32, "BK": 32, "BV": 32}, num_warps=4, num_stages=2),
        triton.Config({"BT": 64, "BK": 16, "BV": 16}, num_warps=4, num_stages=2),
        triton.Config({"BT": 16, "BK": 16, "BV": 16}, num_warps=8, num_stages=2),
        triton.Config({"BT": 32, "BK": 32, "BV": 32}, num_warps=8, num_stages=2),
        triton.Config({"BT": 64, "BK": 16, "BV": 16}, num_warps=8, num_stages=2)
    ],
    key=["T", "NV", "NK", "B", "H"]
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q, k, v, h, g, 
    do, dh,
    dg, 
    dq, dk, 
    s_qk_h, s_qk_t, s_qk_d, 
    s_vo_h, s_vo_t, s_vo_d,
    scale, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    T: tl.constexpr, 
    V: tl.constexpr, 
    H: tl.constexpr, 
    CHECK: tl.constexpr
):
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_h = tl.program_id(2)
    NT = tl.cdiv(T, BT)
    i_b = tl.program_id(3)
    N = tl.num_programs(3)

    p_q = tl.make_block_ptr(
        base=q + i_b * s_qk_h,
        shape=(T, V),
        strides=(s_qk_t, s_qk_d),
        offsets=(0, i_k * BK + i_v * BV),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    p_do = tl.make_block_ptr(
        base=do + i_b * s_vo_h,
        shape=(T, V),
        strides=(s_vo_t, s_vo_d),
        offsets=(0, i_k * BK + i_v * BV),
        block_shape=(BT, BV),
        order=(1, 0)
    )

    p_dh = tl.make_block_ptr(
        base=dh + i_b * s_vo_h,
        shape=(T, V),
        strides=(s_vo_t, s_vo_d),
        offsets=(0, i_k * BK + i_v * BV),
        block_shape=(BT, BV),
        order=(1, 0)
    )

    p_k = tl.make_block_ptr(
        base=k + i_h * s_qk_h,
        shape=(V, T),
        strides=(s_qk_d, s_qk_t),
        offsets=(i_k * BK + i_v * BV, 0),
        block_shape=(BV, BT),
        order=(0, 1)
    )
    p_v = tl.make_block_ptr(
        base=v + i_h * s_vo_h,
        shape=(T, V),
        strides=(s_vo_t, s_vo_d),
        offsets=(0, i_k * BK + i_v * BV),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    b_dq = tl.zeros([BK, BT], dtype=tl.float32)
    b_dk = tl.zeros([BK, BT], dtype=tl.float32)
    b_dg = tl.zeros([BK, BT], dtype=tl.float32)
    _g = tl.full([BK, BT], 1.0 - scale, tl.float32)
    _h = tl.load(h, boundary_check=(0, 2, 1))
    for i in range(NT):
        if CHECK:
            p_v = tl.advance(p_v, (BT, 0))
            p_do = tl.advance(p_do, (BT, 0))
            p_k = tl.advance(p_k, (0, BT))
        else:
            p_dg = tl.make_block_ptr(
                dg + i_b * s_qk_h + i_h * s_qk_h * N,
                (T, V),
                (s_qk_t, s_qk_d),
                (i * BT, i_k * BK + i_v * BV),
                (BT, BV),
                (1, 0)
            )
            _g += tl.load(p_dg, boundary_check=(0, 1), eviction_policy='evict_last')

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        if i > 0 and i < NT - 1:
            _g_bef = tl.full([BK, BT], 1.0 - scale, tl.float32)
            _h_bef = tl.load(h + i_b * V + (i - 1) * BT, boundary_check=(0,))

            m_bef = (i_k * BK + i_v * BV)[None, :] >= 0
            _g_bef = tl.where(m_bef, _g_bef, 0).to(tl.float32)

            b_dq += tl.dot(_h_bef, tl.trans(_h) * _g_bef, allow_tf32=False).to(b_do.dtype)
            b_dk += tl.dot(_h_bef * (_1 - scale), tl.trans(b_v) * _g_bef, allow_tf32=False).to(b_do.dtype)
            b_dg += tl.dot(b_do, tl.trans(_h_bef) * (1 - _g_bef), allow_tf32=False).to(b_do.dtype)
            _g *= _g_bef
            p_do = tl.advance(p_do, (BT, 0))
            p_v = tl.advance(p_v, (BT, 0))
            p_k = tl.advance(p_k, (0, BT))
        else:
            if i == 0:
                _g = tl.minimum(_g, 1.0 - scale)
                p_do = tl.advance(p_do, (BT, 0))
                p_v = tl.advance(p_v, (BT, 0))
            _g = tl.where((i_k * BK + i_v * BV)[None, :] >= 0, _g, 0).to(tl.float32)
            if i >
