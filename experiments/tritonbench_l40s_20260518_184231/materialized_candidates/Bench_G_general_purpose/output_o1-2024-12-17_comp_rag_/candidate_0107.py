import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# rbe_triton kernel
# ------------------------------------------------------------------------------
@triton.jit
def rbe_triton_kernel(
    x_ptr, out_ptr,
    stride_xb, stride_xm, stride_xk,
    stride_outb, stride_outm, stride_outk,
    B, M, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # program IDs for M-, K-/batch- dimensions
    pid_m = tl.program_id(0)
    pid_bk = tl.program_id(1)
    
    # offsets in the M dimension
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # number of blocks covering the K dimension
    num_k_blocks = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # batch ID and K-block ID
    b_id = pid_bk // num_k_blocks
    pid_k = pid_bk % num_k_blocks
    
    # offsets in the K dimension
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # create masks for valid indices
    mask_m = offs_m < M
    mask_k = offs_k < K

    # compute even/odd index (real/imag) positions along last dimension
    real_idx = offs_k * 2
    imag_idx = offs_k * 2 + 1

    # check valid ranges for storing real/imag
    mask_kreal = real_idx < (2 * K)
    mask_kimag = imag_idx < (2 * K)

    # pointers for real and imag components in x
    x_real_ptr = x_ptr + b_id * stride_xb + offs_m[:, None] * stride_xm + real_idx[None, :]
    x_imag_ptr = x_ptr + b_id * stride_xb + offs_m[:, None] * stride_xm + imag_idx[None, :]

    # load real and imag data
    real = tl.where(
        mask_m[:, None] & mask_k[None, :] & mask_kreal[None, :],
        tl.load(x_real_ptr),
        0.0
    )
    imag = tl.where(
        mask_m[:, None] & mask_k[None, :] & mask_kimag[None, :],
        tl.load(x_imag_ptr),
        0.0
    )

    # compute frequency values (sin and cos) for positional transform
    off_k_f = offs_k.to(tl.float32)
    freq_sin = tl.sin(off_k_f / 10000.0)[None, :]
    freq_cos = tl.cos(off_k_f / 10000.0)[None, :]

    # transform
    out_real = real * freq_cos - imag * freq_sin
    out_imag = real * freq_sin + imag * freq_cos

    # pointers for real and imag components in out
