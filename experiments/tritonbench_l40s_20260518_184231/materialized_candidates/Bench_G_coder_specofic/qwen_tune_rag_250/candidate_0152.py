({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_hgrn_fwd_kernel_h(
    x,
    g,
    gc,
    o,
    h0,
    s_xd, s_gb, s_gd, s_od, s_hd,
    T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, USE_INITIAL_STATE: tl.constexpr
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BT, BD], dtype=tl.float32)

    offt_d = i_t * BT + tl.arange(0, BT)
    offt_t = i_t * BT
    offt_bn = i_bh * tl.num_programs(0)
    offt_h = i_d * BD

    mask_d = offt_d < T
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + offt_bn * D + offt_h + tl.arange(0, BD))
    for i in range(0, tl.cdiv(D, BD)):
        p_x = tl.load(x + (offf_t + tl.arange(0, BT))[:, None] * s_xd + (offft_h + i * BD + tl.arange(0, BD))[None, :] + i_bh * s_hd, mask=mask_d[:, None], other=0.0)
        p_g = tl.load(g + offt_bn * s_gb + (offft_h + i * BD + tl.arange(0, BD))[None, :] + i_t * s_gd, mask=mask_d[:, None], other=0.0)
        p_gc = tl.load(gc + offt_bn * s_gb + (offft_h + i * BD + tl.arange(0, BD))[None, :] + i_t * s_gd, mask=mask_d[:, None], other=0.0)
        b_h = b_h * p_gc * tl.exp(p_g - p_gc)[:, None] + p_x * tl.exp(p_g)
        tl.store(o + offt_bn * s_od + offt_d[:, None] * D + offt_h + i * BD + tl.arange(0, BD))[None, :], b_h * tl.exp(-p_gc)[:, None], mask=mask_d[:, None])
    tl.debug_barrier()
    if i_t == tl.cdiv(T, BT) - 1:
        tl.store(h0 + (offft_bn + tl.arange(0, BT))[:, None] * D + (offft_h + tl.arange(0, BD))[None, :] + i_bh * s_hd, b_h * tl.exp(-p_gc)[:, None], mask=(offft_h + tl.arange(0, BD) < D)[:, None])

@triton.autotune(
    configs=[
        triton.Config({'BT': 32}, num_warps=1),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=1),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
        triton.Config({'BT': 128}, num_warps=1),
        triton.Config({'BT': 128}, num_warps=2),
        triton.Config({'BT': 128}, num_warps=4),
        triton.Config({'BT': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_hgrn_fwd_kernel_o(
    g,
    gc,
    o,
    A,
    s_gd, s_od, s_ad, T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    b_o = tl.zeros([BT, 1], dtype=tl.float32)
    b_s = tl.zeros([BT, 1], dtype=tl.float32)
    mask_d = tl.arange(0, BT) < T
    for i in range(0, tl.cdiv(D, i_d * BT)):
        p_g = tl.load(g + i_bh * s_gd + i_d * BT + tl.arange(0, BT), mask=mask_d, other=0.0)
        p_gc = tl.load(gc + i_bh * s_gd + i_d * BT + tl.arange(0, BT), mask=mask_d, other=0.0)
        p_A = tl.load(A + i * BT * D + tl.arange(0, BT)[:, None] * D + tl.arange(0, BT)[None, :], mask=(tl.arange(0, BT)[:, None] < T) & (tl.arange(0, BT)[None, :] < T - i * BT), other=0.0)
        p_o = tl.load(o + i_bh * s_od + i_d * BT * D + tl.arange(0, BT)[:, None] * D + tl.arange(0, BT)[None, :], mask=(tl.arange(0, BT)[:, None] < T) & (tl.arange(0, BT)[None, :] < T - i * BT), other=0.0)
        b_o = b_o * tl.exp(p_gc)[:, None] + tl.sum(p_A * tl.exp(p_g)[:, None] * p_o, 0)[:, None]
        b_s = b_s * tl.exp(p_gc)[:, None] + tl.sum(p_A * tl.exp(p_g)[:, None], 0)[:, None]
    tl.store(A + i_bh * s_ad + i_d * BT + tl.arange(0, BT), b_s, mask=mask_d)
    tl.store(o + i_bh * s_od + i_d * BT * D + tl.arange(0, BT) * D, b_o * tl.exp(-b_s), mask=mask_d)

@triton.autotune(
    configs=[
        triton.Config({'BT': 32}, num_warps=1),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=1),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
        triton.Config({'BT': 128}, num_warps=1),
        triton.Config({'BT': 128}, num_warps=2),
        triton.Config({'BT': 128}, num_warps=4),
        triton.Config({'BT': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_hgrn_bwd_kernel_h(
    x,
    g,
    A,
    dh,
    dgc,
    s_xd, s_gd, s_ad, s_dhd, T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BT, BD], dtype=tl.float32)
    mask_d = i_t * BT + tl.arange(0, BT) < T
    for _ in range(0, tl.cdiv(T, BT)):
        p_x = tl.load(x + (i_t * BT + tl.arange(0, BT))[:, None] * s_xd + (i_d * BD + tl.arange(0, BD))[None, :] + i_bh * D, mask=mask_d[:, None], other=0.0)
        p_g = tl.load(g + i_bh * s_gd + (i_d * BD + tl.arange(0, BD))[None, :] + i_t * BT, mask=mask_d[:, None], other=0.0)
        p_A = tl.load(A + i_t * BT + tl.arange(0, BT)[:, None] * D + (i_d * BD + tl.arange(0, BD))[None, :], mask=mask_d[:, None], other=0.0)
        b_dh = b_dh * tl.exp(-p_g)[:, None] + p_A * tl.exp(p_g)[:, None] * b_dh * tl.exp(-p_g)[:, None]
        tl.store(dh
