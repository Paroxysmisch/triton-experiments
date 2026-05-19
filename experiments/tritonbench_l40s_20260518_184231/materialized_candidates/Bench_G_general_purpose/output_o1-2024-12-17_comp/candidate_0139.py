import triton
import triton.language as tl

class Config:
    def __init__(self, num_warps=4, num_stages=2, num_ctas=128):
        self.num_warps  = num_warps
        self.num_stages = num_stages
        self.num_ctas   = num_ctas

@triton.jit
def matmul_kernel_with_block_pointers(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    strideAm, strideAk,
    strideBm, strideBk,
    strideCm, strideCk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # Grouped ordering
    width = (M + BLOCK_M - 1) // BLOCK_M
    group_id = pid // GROUP_SIZE_M
    group_size = min(width - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % GROUP_SIZE_M)
    pid_n = tl.program_id(axis=1)

    # Compute block-row and block-col of C
    m_block = pid_m * BLOCK_M
    n_block = pid_n * BLOCK_N

    # Create an accumulator in int32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Block pointers for A and B
    a_ptrs = A_ptr + (m_block + tl.arange(0, BLOCK_M)[:, None]) * strideAm \
                     + tl.arange(0, BLOCK_K)[None, :] * strideAk
    b_ptrs = B_ptr + (tl.arange(0, BLOCK_K)[:, None]) * strideBm \
                     + (n_block + tl.arange(0, BLOCK_N)[None, :]) * strideBk

    # Loop over K dimension
    # Each iteration processes a chunk of BLOCK_K
    for k_off in range(0, K, BLOCK_K):
        # Load A and B
        a_tile = tl.load(a_ptrs, mask=(m_block + tl.arange(0, BLOCK_M)[:, None] < M) & (k_off + tl.arange(0, BLOCK_K)[None, :] < K), other=0)
        b_tile = tl.load(b_ptrs, mask=(k_off + tl.arange(0, BLOCK_K)[:, None] < K) & (n_block + tl.arange(0, BLOCK_N)[None, :] < N), other=0)

        # Compute partial matmul
        accumulator += tl.sum(a_tile * b_tile, axis=1)

        # Advance pointers
        a_ptrs += BLOCK_K * strideAk
        b_ptrs += BLOCK_K * strideBm

    # Store results back to C
    c_ptrs = C_ptr + (m_block + tl.arange(0, BLOCK_M)[:, None]) * strideCm \
                     + (n_block + tl.arange(0, BLOCK_N)[None, :]) * strideCk
    mask_c = (m_block + tl.arange(0, BLOCK_M)[:, None] < M) & (n_block + tl.arange(0, BLOCK_N)[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask_c)

@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    A_ptr, B_ptr, scales1_ptr, C_ptr,
    M, N, K,
    strideAm, strideAk,
    strideBm, strideBk,
    strideScales,
    strideCm, strideCk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, EVEN_K: bool
):
    pid = tl.program_id(axis=0)
    width = (M + BLOCK_M - 1) // BLOCK_M
    group_id = pid // GROUP_SIZE_M
    group_size = min(width - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % GROUP_SIZE_M)
    pid_n = tl.program_id(axis=1)

    m_block = pid_m * BLOCK_M
    n_block = pid_n * BLOCK_N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    a_ptrs = A_ptr + (m_block + tl.arange(0, BLOCK_M)[:, None]) * strideAm \
                     + tl.arange(0, BLOCK_K)[None, :] * strideAk
    b_ptrs = B_ptr + (tl.arange(0, BLOCK_K)[:, None]) * strideBm \
                     + (n_block + tl.arange(0, BLOCK_N)[None, :]) * strideBk

    for k_off in range(0, K, BLOCK_K):
        a_block = tl.load(a_ptrs, mask=(m_block + tl.arange(0, BLOCK_M)[:, None] < M) & (k_off + tl.arange(0, BLOCK_K)[None, :] < K), other=0)
        b_block = tl.load(b_ptrs, mask=(k_off + tl.arange(0, BLOCK_K)[:, None] < K) & (n_block + tl.arange(0, BLOCK_N)[None, :] < N), other=0)
        accumulator += tl.sum(a_block * b_block, axis=1)
        a_ptrs += BLOCK_K * strideAk
        b_ptrs += BLOCK_K * strideBm

    # Load scales and apply
    s_ptrs = scales1_ptr + (m_block + tl.arange(0, BLOCK_M)[:, None]) * strideScales
    scales = tl.load(s_ptrs, mask=(m_block + tl.arange(0, BLOCK_M)[:, None] < M), other=0)
    # Broadcast scales along BLOCK_N dimension
    scales = tl.broadcast_to(scales, (BLOCK_M, BLOCK_N))
    accumulator = accumulator * scales

    # Store to C
    c_ptrs = C_ptr + (m_block + tl.arange(0, BLOCK_M)[:, None]) * strideCm \
                     + (n_block + tl.arange(0, BLOCK_N)[None, :]) * strideCk
    mask_c = (m_block + tl.arange(0, BLOCK_M)[:, None] < M) & (n_block + tl.arange(0, BLOCK_N)[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask_c)

def int_matmul_kernel(a, b, c, config: Config):
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Incompatible matrix dimensions"
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = lambda meta: (
        ( (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'] ) * meta['GROUP_SIZE_M'],
        ( (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'] )
    )
    matmul_kernel_with_block_pointers[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=8,
        num_warps=config.num_warps, 
        num_stages=config.num_stages
    )

def int_scaled_matmul_kernel(a, b, scales1, c, config: Config):
    M, K = a.shape
    Kb, N = b.shape
    Ms = scales1.shape[0]
    assert K == Kb, "Incompatible matrix dimensions"
    assert M == Ms, "Incompatible scale dimension"
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = lambda meta: (
        ( (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'] ) * meta['GROUP_SIZE_M'],
        ( (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'] )
    )
    scaled_matmul_kernel_with_block_pointers[grid](
        a, b, scales1, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        scales1.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=8, 
        EVEN_K=(K % BLOCK_K == 0),
        num_warps=config.num_warps, 
        num_stages=config.num_stages
    )
