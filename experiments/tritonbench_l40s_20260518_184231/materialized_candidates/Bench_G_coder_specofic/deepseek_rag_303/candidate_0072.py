import triton
import triton.language as tl

@triton.jit(do_not_specialize=["qk_log2_scale", "qk_int_scale"], do_not_specialize_in_device=True)
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q, k, g, A, b_A, g_last, qk_log2_scale, qk_int_scale,
    start_g, end_g, start_n, end_n, start_b, end_b,
    N, s_n_b, S_b,
    stride_qg, stride_qh, stride_qm, stride_kg, stride_kh, stride_kn,
    stride_gm, stride_gn, stride_am, stride_an, stride_b_Am, stride_b_An,
    stride_g_lastm, stride_g_lastn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_M: tl.constexpr,
):
    # Kernel implementation

@triton.jit(do_not_specialize=["qk_log2_scale", "qk_int_scale"], do_not_specialize_in_device=True)
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q, k, g, A, b_A, g_last, qk_log2_scale, qk_int_scale,
    start_g, end_g, start_n, end_n, start_b, end_b,
    N, s_n_b, S_b,
    stride_qg, stride_qh, stride_qm, stride_kg, stride_kh, stride_kn,
    stride_gm, stride_gn, stride_am, stride_an, stride_b_Am, stride_b_An,
    stride_g_lastm, stride_g_lastn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_M: tl.constexpr,
):
    # Kernel implementation

@triton.jit(do_not_specialize=["qk_log2_scale", "qk_int_scale"], do_not_specialize_in_device=True)
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q, k, g, A_intra, g_last, qk_log2_scale, qk_int_scale,
    start_g, end_g, start_n, end_n, start_b, end_b,
    N, s_n_b, S_b,
    stride_qg, stride_qh, stride_qm, stride_kg, stride_kh, stride_kn,
    stride_gm, stride_gn, stride_am, stride_an, stride_g_lastm, stride_g_lastn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL_M: tl.constexpr, SPLIT_K: tl.constexpr,
):
    # Kernel implementation

@triton.jit(do_not_specialize=["qk_log2_scale", "qk_int_scale"])
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A, A_intra,
    start_g, end_g, start_n, end_n, start_b, end_b,
    N, s_n_b, S_b, stride_am, stride_an, stride_A_intram, stride_A_intran,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Kernel implementation

@triton.jit
def chunk_gla_fwd_kernel_o(
    g, A, o, g_last,
    start_g, end_g, start_n, end_n, start_b, end_b,
    N, s_n_b,
    stride_gg, stride_gh, stride_gm, stride_gn,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_O: tl.constexpr,
    BLOCK_DMODEL_M: tl.constexpr,
):
    # Kernel implementation

@torch.inference_mode()
def chunk_fwd_intra_gated_gk_fn(g, k, g_last, scale, qk_log2_scale, qk_int_scale, o, A, BLOCK_M, BLOCK_N, BLOCK_DMODEL_M, SPLIT_K, max_chunk_size, n_g):
    # Wrapper function implementation

@torch.inference_mode()
def chunk_fwd_o_gated_gk_fn(g, k, g_last, scale, qk_log2_scale, qk_int_scale, o, A, BLOCK_M, BLOCK_N, BLOCK_DMODEL_M, BLOCK_DMODEL_O, SPLIT_K, max_chunk_size, n_g):
    # Wrapper function implementation
