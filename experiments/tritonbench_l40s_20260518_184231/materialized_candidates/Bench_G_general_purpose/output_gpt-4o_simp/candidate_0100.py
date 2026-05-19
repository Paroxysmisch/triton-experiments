import triton
import triton.language as tl

# Define the Triton kernel for matrix multiplication with optional leaky ReLU activation
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr, ACTIVATION: tl.constexpr):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate block row and column
    block_row = pid_m * BLOCK_SIZE_M
    block_col = pid_n * BLOCK_SIZE_N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B
        a = tl.load(a_ptr + (block_row + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_am + (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * stride_ak)
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * stride_bk + (block_col + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_bn)

        # Matrix multiplication
        acc += tl.dot(a, b)

    # Optional leaky ReLU activation
    if ACTIVATION == 'leaky_relu':
        acc = tl.where(acc > 0, acc, 0.01 * acc)

    # Store result in C
    c = c_ptr + (block_row + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_cm + (block_col + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_cn
    tl.store(c, acc)


def matmul(a, b, activation=None):
    # Extract dimensions
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Incompatible dimensions for matrix multiplication."

    # Strides for input matrices
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()

    # Allocate output matrix
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stride_cm, stride_cn = c.stride()

    # Define grid size
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        ACTIVATION=activation
    )

    return c
