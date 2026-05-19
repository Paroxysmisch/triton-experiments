@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr, mask_ptr, sm_scale,
    m_size, n_size, d_model, num_heads, causal_mask,
    BLOCK_M, BLOCK_N, BLOCK_DMODEL,
    USE_FP8, Lk, num_stages
):
    # Kernel code goes here
