import triton
import triton.language as tl

# Define the kernel function
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_m = pid // (num_pid_n * num_pid_k)
    pid_n = (pid // num_pid_k) % num_pid_n
    pid_k = pid % num_pid_k

    # Coordinates of the current block in the global matrix
    m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Create pointers for A and B
    a_ptrs = a_ptr + m[:, None] * stride_am + k[None, :] * stride_ak
    b_ptrs = b_ptr + k[:, None] * stride_bk + n[None, :] * stride_bn

    # Initialize C
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Compute C
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs + k * stride_ak, mask=k < K, other=0.0)
        b = tl.load(b_ptrs + k * stride_bk, mask=k < K, other=0.0)
        acc += tl.dot(a, b)

    # Write back to C
    c_ptr += m[:, None] * stride_cm + n[None, :] * stride_cn
    tl.store(c_ptr, acc, mask=(m < M)[:, None] & (n < N)[None, :])

# Define the wrapper function
def triton_matmul(a, b):
    # Check dimensions
    assert a.ndim == 2 and b.ndim == 2, "Input matrices must be 2D"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"

    M, K = a.shape
    K, N = b.shape

    # Initialize output matrix
    c = tl.zeros((M, N), dtype=a.dtype)

    # Configure grid and block sizes
    block_size_m = 128
    block_size_n = 128
    block_size_k = 32

    # Calculate number of blocks
    num_blocks_m = (M + block_size_m - 1) // block_size_m
    num_blocks_n = (N + block_size_n - 1) // block_size_n
    num_blocks_k = (K + block_size_k - 1) // block_size_k

    # Launch the kernel
    grid = (num_blocks_m * num_blocks_n * num_blocks_k,)
    matmul_kernel[grid](a.data_ptr(), b.data_ptr(), c.data_ptr(),
                        M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        block_size_m, block_size_n, block_size_k)

    return c
