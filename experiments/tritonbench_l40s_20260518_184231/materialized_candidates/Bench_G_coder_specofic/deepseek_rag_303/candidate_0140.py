import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BT': 16, 'BK': 32}, num_warps=2, num_stages=2),
    ],
    key=['D'],
)
@triton.jit
def kernel_fwd_decay_cumsum(
    g, g_o, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, DK: tl.constexpr, D: tl.constexpr
):
    fwd_decay_cumsum(
        g, g_o, s_qk_h, s_qk_t, s_qk_d,
        B, H, T, scale,
        BT=BT, BK=BK, DK=DK
    )

def launch_kernel_fwd_decay_cumsum(
    g, g_o, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale, BT, BK, DK
):
    grid = (DK // BK, T // BT, B * H)
    kernel_fwd_decay_cumsum[grid](
        g, g_o, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale, BT=BT, BK=BK, DK=DK, D=g.shape[-1]
    )

@triton.autotune(
    configs=[
        triton.Config({'BT': 16, 'BK': 32}, num_warps=4, num_stages=3),
    ],
    key=['D'],
)
@triton.jit
def kernel_prepare_qg_kg(
    q, k, g, qg, kg, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, DK: tl.constexpr, D: tl.constexpr
):
    prepare_qg_kg(
        q, k, g, qg, kg, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
        BT=BT, BK=BK, DK=DK
    )

def launch_kernel_prepare_qg_kg(
    q, k, g, qg, kg, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale, BT, BK, DK
):
    grid = (DK // BK, T // BT, B * H)
    kernel_prepare_qg_kg[grid](
        q, k, g, qg, kg, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale, BT=BT, BK=BK, DK=DK, D=g.shape[-1]
    )

@triton.autotune(
    configs=[
        triton.Config({'BT': 16, 'BK': 32}, num_warps=4, num_stages=5),
    ],
    key=['D'],
)
@triton.jit
def kernel_bwd_decay_global_cumsum(
    dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg,
    s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, DK: tl.constexpr, D: tl.constexpr
):
    bwd_decay_global_cumsum(
        dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg,
        s_qk_h, s_qk_t, s_qk_d,
        B, H, T, scale,
        BT=BT, BK=BK, DK=DK
    )

def launch_bwd_decay_global_cumsum(
    dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, dx_inner,
    s_qk_h, s_qk_t, s_qk_d, B, H, Tl, scale, BT, BK, DK
):
    grid = (DK // BK, Tl // BT, B * H)
    kernel_bwd_decay_global_cumsum[grid](
        dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, s_qk_h,
        s_qk_t, s_qk_d, B, H, Tl, scale,
        BT=BT, BK=BK, DK=DK,  D=g.shape[-1]
    )
