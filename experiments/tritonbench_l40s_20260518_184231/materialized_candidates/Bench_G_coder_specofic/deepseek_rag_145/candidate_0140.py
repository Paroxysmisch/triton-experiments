import triton
import triton.language as tl

inv_ln2 = 1.44269504

# Kernel for forward decay cumulative sum
@triton.jit
def fwd_decay_cumsum(
    g, g_o, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, DK: tl.constexpr
):
    # kernel implementation

# Kernel for preparing qg and kg
@triton.jit
def prepare_qg_kg(
    q, k, g, qg, kg, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, DK: tl.constexpr
):
    # kernel implementation

# Kernel for backward decay global cumulative sum
@triton.jit
def bwd_decay_global_cumsum(
    dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg,
    s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, DK: tl.constexpr
):
    # kernel implementation

def launch_fwd_decay_cumsum(
    g, g_o, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT, BK, DK
):
    grid = lambda M: (triton.next_pow_2(M) // BK, )
    fwd_decay_cumsum[grid(DK)][(BK, 1, 1), (1, 1, 1)](
        g, g_o, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
        BT, BK, DK
    )

def launch_prepare_qg_kg(
    q, k, g, qg, kg, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT, BK, DK
):
    grid = lambda M: (triton.next_pow_2(M) // BK, )
    prepare_qg_kg[grid(DK)][(BK, 1, 1), (1, 1, 1)](
        q, k, g, qg, kg, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
        BT, BK, DK
    )

def launch_bwd_decay_global_cumsum(
    dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg,
    s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT, BK, DK
):
    grid = lambda M: (triton.next_pow_2(M) // BK, )
    bwd_decay_global_cumsum[grid(DK)][(BK, 1, 1), (1, 1, 1)](
        dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg,
        s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
        BT, BK, DK
    )
