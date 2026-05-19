import triton
import triton.language as tl

@triton.jit
def _addmm_kernel(
    alpha,
    beta,
    IS_BETA_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    VALUES_PTR: tl.constexpr,
    VALUES_STRIDE_BATCH: tl.constexpr,
    VALUES_STRIDE_OUT: tl.constexpr,
    MAT1_PTR: tl.constexpr,
    MAT1_STRIDE_BATCH: tl.constexpr,
    MAT1_STRIDE_K: tl.constexpr,
    MAT2_PTR: tl.constexpr,
    MAT2_STRIDE_BATCH: tl.constexpr,
    MAT2_STRIDE_K: tl.constexpr,
    ACCUMULATED_STRIDE_BATCH: tl.constexpr,
    ACCUMULATED_STRIDE_OUT: tl.constexpr,
    DATA_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if pid_m >= grid_m or pid_n >= grid_n:
        return

    accum = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=DATA_TYPE)

    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = k + BLOCK_SIZE_K > K

        mat1_block = tl.load(
            MAT1_PTR + (pid_m * BLOCK_SIZE_M * K + k) * MAT1_STRIDE_K,
            mask=k_mask,
            other=tl.zeros_like(tl.load(MAT1_PTR, mask=False))
        )

        mat2_block = tl.load(
            MAT2_PTR + (k * N + pid_n * BLOCK_SIZE_N) * MAT2_STRIDE_K,
            mask=k_mask,
            other=tl.zeros_like(tl.load(MAT2_PTR, mask=False))
        )

        accum += tl.dot(mat1_block, mat2_block, allow_tf32=ALLOW_TF32)

    out_ptr = VALUES_PTR + (pid_m * BLOCK_SIZE_M + pid_n) * VALUES_STRIDE_OUT
    accum_ptr = ACCUMULATED_STRIDE_BATCH + pid_m * BLOCK_SIZE_M * BLOCK_SIZE_N * VALUES_STRIDE_OUT + pid_n * BLOCK_SIZE_N * VALUES_STRIDE_OUT

    if IS_BETA_ZERO:
        accum *= alpha
    else:
        accum = alpha * accum + beta * tl.load(accum_ptr)

    tl.store(accum_ptr, accum)
