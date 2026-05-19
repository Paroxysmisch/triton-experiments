import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')

@triton.autotune(
    configs=[
        triton.Config({'BT': 64, 'BK': 64, 'BV': 64}, num_warps=8, num_stages=1),
        triton.Config({'BT': 128, 'BK': 64, 'BV': 64}, num_warps=4, num_stages=1),
        triton.Config({'BT': 128, 'BK': 128, 'BV': 64}, num_warps=4, num_stages=1),
        triton.Config({'BT': 128, 'BK': 128, 'BV': 128}, num_warps=4, num_stages=1),
        triton.Config({'BT': 128, 'BK': 64, 'BV': 128}, num_warps=4, num_stages=1),
        triton.Config({'BT': 64, 'BK': 128, 'BV': 64}, num_warps=8, num_stages=1),
        triton.Config({'BT': 64, 'BK': 64, 'BV': 128}, num_warps=8, num_stages=1),
        triton.Config({'BT': 64, 'BK': 128, 'BV': 128}, num_warps=8, num_stages=1),
        triton.Config({'BT': 64, 'BK': 64, 'BV': 64}, num_warps=4, num_stages=1),
        triton.Config({'BT': 64, 'BK': 64, 'BV': 64}, num_warps=2, num_stages=1),
        triton.Config({'BT': 128, 'BK': 64, 'BV': 64}, num_warps=2, num_stages=1),
        triton.Config({'BT': 64, 'BK': 128, 'BV': 64}, num_warps=2, num_stages=1),
        triton.Config({'BT': 64, 'BK': 64, 'BV': 128}, num_warps=2, num_stages=1),
    ],
    key=['n_heads', 'dim', 'n_cols'],
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q, k, v, h, g, o, softmax_o,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_g_h, s_g_t, s_g_d,
    s_o_h, s_o_t, s_o_d,
    s_softmax_o_h, s_softmax_o_t, s_softmax_o_d,
    scale,
    n_cols, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    bk = tl.cdiv(BT, BK)

    # block pointers
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (BT,), (s_qk_t,), (0,), (i_t * BT,), (BT,), (1,))
    chunk_k, chunk_v = tl.make_block_ptr(k, (n_cols * bk,), (s_qk_t, s_qk_d), (0, i_bh * n_cols * bk), (i_t * BT,), (BT,), (1,))

    chunk_o = tl.make_block_ptr(o + i_bh * s_o_h, (BT, n_cols), (s_o_t, s_o_d), (0,), (i_t * BT,), (BT,), (1,))
    chunk_softmax_o = tl.make_block_ptr(softmax_o + i_bh * s_softmax_o_h, (BT, bk), (s_softmax_o_t, s_softmax_o_d), (0,), (i_t * BT,), (BT,), (1,))
    chunk_h = tl.make_block_ptr(h + i_bh * s_vo_h, (BT,), (s_vo_t,), (0,), (i_t * BT,), (BT,), (1,))
    p_h = tl.make_block_ptr(h + i_bh * s_vo_h, (bk * BT,), (s_vo_t,), (0,), (0,), (BT,), (1,))
    chunk_g_ptrs, chunk_g_strides, chunk_g_sizes = tl.generate_chunk_rev_indices(i_bh * n_cols * bk, n_cols * bk, n_cols, bk)
    chunk_g = tl.make_block_ptr(g, (n_cols * bk,), (s_g_t, s_g_d), (0,), chunk_g_ptrs, chunk_g_sizes, chunk_g_strides, contiguous=True)

    # mask
    b_k, b_bs = tl.zeros([BK, BK], dtype=tl.int32), tl.zeros([BK], dtype=tl.int32)
    for i in range(0, bk):
        p_k = tl.advance(chunk_k, (i * BK,))
        m_k = tl.arange(0, BK) < n_cols * bk - i * BK
        tl.store(b_k, tl.load(p_k, boundary_check=(0, m_k)).to(tl.float32) * scale, boundary_check=(0, 1))
        tl.store(b_bs, tl.where(m_k, 1, 0), boundary_check=(0, 1))
        i_bs = tl.arange(0, BK)[None, :] < (i + 1)[:, None] * b_bs[None, :]
        b_bs = tl.maximum(b_bs, i_bs)

    b_s = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(bk):
        p_v = tl.advance(chunk_v, (i * BV,))
        mask = (n_cols * bk - i * BV + tl.arange(0, BV)) < tl.cdiv(BT, BV) * BV
        v = tl.load(p_v, boundary_check=(0, mask))

        p_k = tl.advance(chunk_k, (i * BK,))
        key = tl.load(p_k, boundary_check=(0, True)).to(tl.float32)
        p_o1 = tl.advance(p_q, (0, )), (chunk_o, (0, i * BV,), (s_vo_t, s_vo_d,))
        expo = key - tl.sum(key[:, None] * v, 0)[None, :]

        denom = tl.sum(tl.exp(expo),
