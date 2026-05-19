triton
import triton
import triton.language as tl

# Constants
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 128

@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_scales_g, stride_scales_n, stride_zeros_g, stride_zeros_n,
    groupsize, NO_GROUPS,
    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
):
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    pid_m = pid // (grid_n * grid_k)
    pid_n = (pid // grid_k) % grid_n
    pid_k = pid % grid_k

    # Matrix indices
    m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Clamp indices
    m = tl.min(m, M - 1)
    n = tl.min(n, N - 1)
    k = tl.min(k, K - 1)

    # Load A, B, C
    a = tl.load(a_ptr + m * stride_am + k * stride_ak, mask=m < M and k < K)
    b = tl.load(b_ptr + k * stride_bk + n * stride_bn, mask=k < K and n < N)
    c = tl.load(c_ptr + m * stride_cm + n * stride_cn, mask=m < M and n < N)

    # Load scales and zeros
    scale_g = tl.load(scales_ptr + pid_k * stride_scales_g, mask=pid_k < NO_GROUPS)
    scale_n = tl.load(scales_ptr + pid_n * stride_scales_n, mask=pid_n < NO_GROUPS)
    zero_g = tl.load(zeros_ptr + pid_k * stride_zeros_g, mask=pid_k < NO_GROUPS)
    zero_n = tl.load(zeros_ptr + pid_n * stride_zeros_n, mask=pid_n < NO_GROUPS)

    # Dequantize B
    b_dequant = b * scale_g + zero_g

    # Initialize result
    c_out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    # Matrix multiplication
    for i in range(BLOCK_SIZE_K):
        a_elem = a[i]
        b_elem = b_dequant[i]
        c_out += a_elem * b_elem

    # Write result
    tl.store(c_ptr + m * stride_cm + n * stride_cn, c_out, mask=m < M and n < N)

@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
    K, N, group_size,
    stride_bk, stride_bn, stride_bsk, stride_bsn, stride_bzpk, stride_bzpn, stride_fpbk, stride_fpbn,
    BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N
):
    pid = tl.program_id(axis=0)
    grid_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_k = pid // grid_n
    pid_n = pid % grid_n

    # Matrix indices
    k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Clamp indices
    k = tl.min(k, K - 1)
    n = tl.min(n, N - 1)

    # Load B, scales, zeros
    b = tl.load(b_ptr + k * stride_bk + n * stride_bn, mask=k < K and n < N)
    scale = tl.load(b_scale_ptr + k * stride_bsk + n * stride_bsn, mask=k < K and n < N)
    zero = tl.load(b_zp_ptr + k * stride_bzpk + n * stride_bzpn, mask=k < K and n < N)

    # Dequantize B
    b_dequant = b * scale + zero

    # Store result
    tl.store(fpb_ptr + k * stride_fpbk + n * stride_fpbn, b_dequant, mask=k < K and n < N)

# Wrapper functions
@triton.jit
def dequantize_int4(b_ptr, b_scale_ptr, b_zp_ptr, K, N, group_size, stride_bk, stride_bn, stride_bsk, stride_bsn, stride_bzpk, stride_bzpn, stride_fpbk, stride_fpbn):
    fpb_ptr = tl.zeros((K, N), dtype=tl.float16)
    dequantize_kernel[binding={0: b_ptr, 1: b_scale_ptr, 2: b_zp_ptr, 3: fpb_ptr}, grid=(K * N + BLOCK_SIZE_K * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_K * BLOCK_SIZE_N)](
        b_ptr, b_scale_ptr, b_zp_ptr, fpb_ptr,
        K, N, group_size,
        stride_bk, stride_bn, stride_bsk, stride_bsn, stride_bzpk, stride_bzpn, stride_fpbk, stride_fpbn
    )
    return fpb_ptr

@triton.jit
def matmul_dequantize_int4_s1(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales_g, stride_scales_n, stride_zeros_g, stride_zeros_n, groupsize, NO_GROUPS):
    fpb_ptr = dequantize_int4(b_ptr, scales_ptr, zeros_ptr, K, N, groupsize, stride_bk, stride_bn, stride_scales_g, stride_scales_n, stride_zeros_g, stride_zeros_n, stride_cm, stride_cn)
    matmul4_kernel[a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales_g, stride_scales_n, stride_zeros_g, stride_zeros_n, groupsize, NO_GROUPS](
        a_ptr, fpb_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales_g, stride_scales_n, stride_zeros_g, stride_zeros_n, groupsize, NO_GROUPS
    )
