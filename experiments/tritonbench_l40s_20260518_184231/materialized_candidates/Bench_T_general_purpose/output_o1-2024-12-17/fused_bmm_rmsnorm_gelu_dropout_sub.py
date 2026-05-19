import torch
import triton
import triton.language as tl


# -----------------------------------------
# Triton kernel: fused bmm + rmsnorm + gelu + dropout + sub
# -----------------------------------------
@triton.jit
def _fused_bmm_rmsnorm_gelu_dropout_sub_kernel(
    # Pointers
    X_ptr,  # float32
    Y_ptr,  # float32
    O_ptr,  # float32
    OUT_ptr,  # float32
    # Strides
    stride_xb, stride_xn, stride_xm,
    stride_yb, stride_ym, stride_yp,
    stride_ob, stride_on, stride_op,
    stride_outb, stride_outn, stride_outp,
    # Dimensions
    B, N, M, P,
    # Extra params
    dropout_p, training, approximate, eps,
    seed,  # for dropout RNG
    # Meta
    BLOCK_M: tl.constexpr,  # block size for N dimension
    BLOCK_N: tl.constexpr,  # block size for P dimension
    BLOCK_K: tl.constexpr,  # block size for M dimension
):
    """
    Each program instance computes a [BLOCK_M, BLOCK_N] tile of the final output
    across the batch dimension as well. We use a 3D launch grid:
      - grid(0): B
      - grid(1): ceil_div(N, BLOCK_M)
      - grid(2): ceil_div(P, BLOCK_N)
    """

    # Program IDs.
    b_id = tl.program_id(0)       # Which batch
    n_block_id = tl.program_id(1) # Which block along the N dimension
    p_block_id = tl.program_id(2) # Which block along the P dimension

    # Starting indices for the block in output space
    n_start = n_block_id * BLOCK_M
    p_start = p_block_id * BLOCK_N

    # Create a range of offsets for N and P within the block
    rn = n_start + tl.arange(0, BLOCK_M)
    rp = p_start + tl.arange(0, BLOCK_N)

    # Create a pointer offset for reading/writing
    # We also clamp the range to avoid out-of-bounds
    rn_cl = tl.where(rn < N, rn, N - 1)
    rp_cl = tl.where(rp < P, rp, P - 1)

    # -------------------------------
    # 1) Compute partial matmul for tile
    # -------------------------------
    # Accumulator for [BLOCK_M, BLOCK_N]
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K blocks
    # We'll compute partial dot-product along M dimension.
    # K in [0 .. M) in steps of BLOCK_K
    num_k_blocks = (M + BLOCK_K - 1) // BLOCK_K

    for k_block_id in range(num_k_blocks):
        k_start = k_block_id * BLOCK_K
        rk = k_start + tl.arange(0, BLOCK_K)
        rk_cl = tl.where(rk < M, rk, M - 1)

        # Load X tile of shape [BLOCK_M, BLOCK_K]
        # X index: b_id, rn_cl, rk_cl
        # pointer offset = b_id*stride_xb + rn_cl*stride_xn + rk_cl*stride_xm
        x_ptrs = X_ptr + (b_id * stride_xb \
                          + rn_cl[:, None] * stride_xn \
                          + rk_cl[None, :] * stride_xm)

        # Load Y tile of shape [BLOCK_K, BLOCK_N]
