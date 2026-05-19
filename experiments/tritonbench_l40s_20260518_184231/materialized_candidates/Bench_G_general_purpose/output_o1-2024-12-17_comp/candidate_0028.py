import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_bx, stride_sx, stride_hx, stride_kx,
    stride_bc, stride_sc, stride_hc, stride_kc,
    stride_bs, stride_ss, stride_hs, stride_ks,
    B, S, H, K,
    CONJUGATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    m = tl.arange(0, BLOCK_M) + tl.program_id(0) * BLOCK_M
    n = tl.arange(0, BLOCK_K) + tl.program_id(1) * BLOCK_K
    m_bound = B * H * (S if not IS_VARLEN else 1)
    mask_m = m < m_bound
    k_bound = K if not INTERLEAVED else K // 2
    mask_n = n < k_bound
    m = tl.where(mask_m, m, 0)
    n = tl.where(mask_n, n, 0)
    b = m // (H * (S if not IS_VARLEN else 1))
    hs = m % (H * (S if not IS_VARLEN else 1))
    h = hs // (S if not IS_VARLEN else 1)
    s = hs % (S if not IS_VARLEN else 1)
    if INTERLEAVED:
        real_off = n * 2
        imag_off = real_off + 1
        x_real = tl.load(x_ptr + b*stride_bx + s*stride_sx + h*stride_hx + real_off*stride_kx, mask=mask_m & mask_n, other=0.0)
        x_imag = tl.load(x_ptr + b*stride_bx + s*stride_sx + h*stride_hx + imag_off*stride_kx, mask=mask_m & mask_n, other=0.0)
        cos_val = tl.load(cos_ptr + b*stride_bc + s*stride_sc + h*stride_hc + n*stride_kc, mask=mask_m & mask_n, other=0.0)
        sin_val = tl.load(sin_ptr + b*stride_bs + s*stride_ss + h*stride_hs + n*stride_ks, mask=mask_m & mask_n, other=0.0)
        if CONJUGATE:
            out_real = x_real * cos_val + x_imag * sin_val
            out_imag = -x_real * sin_val + x_imag * cos_val
        else:
            out_real = x_real * cos_val - x_imag * sin_val
            out_imag = x_real * sin_val + x_imag * cos_val
        tl.store(out_ptr + b*stride_bx + s*stride_sx + h*stride_hx + real_off*stride_kx, out_real, mask=mask_m & mask_n)
        tl.store(out_ptr + b*stride
