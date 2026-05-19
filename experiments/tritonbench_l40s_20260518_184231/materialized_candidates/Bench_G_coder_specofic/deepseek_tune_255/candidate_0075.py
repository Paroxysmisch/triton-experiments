import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 16}, num_warps=8),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
        triton.Config({'BT': 128}, num_warps=2),
        triton.Config({'BT': 128}, num_warps=4),
        triton.Config({'BT': 128}, num_warps=8),
        triton.Config({'BT': 256}, num_warps=2),
        triton.Config({'BT': 256}, num_warps=4),
        triton.Config({'BT': 256}, num_warps=8),
    ],
    key=['BT'],
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q, k, v, h, g, o, s_q_h, s_q_t, s_q_d, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, T, K, V, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, scale: tl.constexpr,
):
    i_v, i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    i_bh % (BK * BV)
    o_i = i_c * s_q_t + i_bh * s_q_h + tl.arange(0, BT)
    m_s = tl.arange(0, BT) < T
    p_q = tl.make_block_ptr(q + i_bh * s_q_h, (T, K), (s_q_t, s_q_d), (0, i_k * BT), (BT, 1), (1, 0))
    p_k = tl.make_block_ptr(k + i_k * s_k_h, (K, V), (s_k_t, s_k_d), (0, i_v * BV), (1, 1), (1, 0))
    p_v = tl.make_block_ptr(v + i_v * s_v_h, (K, V), (s_v_t, s_v_d), (i_k * BT, 0), (1, BT), (BT, 1))
    p_h = tl.make_block_ptr(h + i_bh * s_h_h, (T, K), (s_h_t, s_h_d), (0, i_k * BT), (BT, 1), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_q_h, (T, K), (s_q_t, s_q_d), (0, i_k * BT), (BT, 1), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BT, BT], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BV)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_o += tl.dot(b_q, (b_k * b_v).to(b_q.dtype), allow_tf32=False)
        b_s += tl.exp(b_h) * b_g
        p_k = tl.advance(p_k, (0, BV))
        p_v = tl.advance(p_v, (BV, 0))
        p_h += (BV * BT)
        p_g += (BV * BT)
    tl.store(o + o_i, tl.sum(b_o, 1)[None, :], boundary_check=(0,))
    tl.store(o + o_i + s_q_t * T, tl.sum(tl.exp(p_h + (BV - 1) * BT) * b_g, 1)[None, :], boundary_check=(0,))

@torch.no_grad()
def chunk_fwd_o_fn(q, k, v, h, g, scale, BK, BV):
    BT = q.shape[1] // BK
    o = torch.empty_like(q)
    grid = lambda META: (META['K'], META['C'], META['BK'], META['BV'])
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        BT, K=k.shape[0], V=v.shape[0], BK=BK, BV=BV, scale=scale
    )
    return o
