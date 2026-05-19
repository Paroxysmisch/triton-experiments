import triton
import triton.language as tl

# Constants for block sizes
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 8

# Define the kernel with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 8}, num_stages=2, num_warps=4),
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
    leaky_relu_alpha,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Matrix indices
    pid = tl.program_id(axis=0)
    pid_m = pid // (BLOCK_SIZE_N * BLOCK_SIZE_M)
    pid_n = pid % (BLOCK_SIZE_N * BLOCK_SIZE_M) // BLOCK_SIZE_M
    pid_k = pid % (BLOCK_SIZE_N * BLOCK_SIZE_M) % BLOCK_SIZE_M

    # Block indices
    bm = pid_m * BLOCK_SIZE_M
    bn = pid_n * BLOCK_SIZE_N
    bk = pid_k * BLOCK_SIZE_K

    # Matrix coordinates
    m = bm + tl.arange(0, BLOCK_SIZE_M)
    n = bn + tl.arange(0, BLOCK_SIZE_N)
    k = bk + tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Load A and B elements
    a = tl.load(a_ptr + m[:, None] * stride_am + k[None, :] * stride_ak)
    b = tl.load(b_ptr + k[:, None] * stride_bk + n[None, :] * stride_bn)

    # Compute dot product
    for k_idx in range(0, K, BLOCK_SIZE_K):
        acc += a * b
        a = tl.load(a_ptr + m[:, None] * stride_am + (k_idx + BLOCK_SIZE_K) * stride_ak)
        b = tl.load(b_ptr + (k_idx + BLOCK_SIZE_K) * stride_bk + n[None, :] * stride_bn)

    # Apply leaky ReLU if alpha is not zero
    if leaky_relu_alpha != 0.0:
        acc = tl.where(acc < 0.0, acc * leaky_relu_alpha, acc)

    # Write result to C
    c = acc.to(tl.float16)
    tl.store(c_ptr + m[:, None] * stride_cm + n[None, :] * stride_cn, c)

# Define the main function to call the kernel
def matmul(a, b, leaky_relu_alpha=0.0):
    M, K = a.shape
    K, N = b.shape

    # Validate input tensors
    assert a.is_contiguous(), "Input tensor a must be contiguous"
    assert b.is_contiguous(), "Input tensor b must be contiguous"
    assert a.dtype == tl.float16, "Input tensor a must be of type float16"
    assert b.dtype == tl.float16, "Input tensor b must be of type float16"

    # Allocate output tensor
    c = tl.zeros((M, N), dtype=tl.float16)

    # Compute grid and block sizes
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )

    # Launch the kernel
    matmul_kernel[grid](a.data_ptr(), b.data_ptr(), c.data_ptr(),
                        M, N, K,
                        a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        leaky_relu_alpha)

    return c
