triton
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    a_shape, b_shape, c_shape,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    stride_am: tl.constexpr, stride_an: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    num_stages: tl.constexpr, num_warps: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = c_shape[0] // BLOCK_SIZE_M
    grid_n = c_shape[1] // BLOCK_SIZE_N
    grid_k = c_shape[2] // BLOCK_SIZE_K

    m = pid // (grid_n * grid_k)
    n = (pid // grid_k) % grid_n
    k = pid % grid_k

    # Compute block offsets
    a_offs = m * BLOCK_SIZE_M * stride_am + k * BLOCK_SIZE_K * stride_ak
    b_offs = k * BLOCK_SIZE_K * stride_bk + n * BLOCK_SIZE_N * stride_bn
    c_offs = m * BLOCK_SIZE_M * stride_cm + n * BLOCK_SIZE_N * stride_cn

    # Allocate registers for matrix sub-matrices
    a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Load sub-matrices with masking for boundary conditions
    for k_idx in range(0, BLOCK_SIZE_K, BLOCK_SIZE_K // num_stages):
        a += tl.load(a_ptr + a_offs + k_idx * stride_ak, mask=(k_idx < BLOCK_SIZE_K), eviction_policy=tl.core.EvictionPolicy.LRU)
        b += tl.load(b_ptr + b_offs + k_idx * stride_bk, mask=(k_idx < BLOCK_SIZE_K), eviction_policy=tl.core.EvictionPolicy.LRU)

    # Perform matrix multiplication
    for k_idx in range(0, BLOCK_SIZE_K, BLOCK_SIZE_K // num_stages):
        c += tl.dot(a, b)

    # Store results to global memory
    tl.store(c_ptr + c_offs, c, mask=(m < grid_m and n < grid_n))
