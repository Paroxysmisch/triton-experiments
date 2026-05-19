import triton
import triton.language as tl

# Define the matrix multiplication kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    apply_leaky_relu: tl.constexpr,
    alpha: tl.constexpr
):
    # Compute the grid and block indices
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_warp = num_pid_m * num_pid_n
    num_warp_m = tl.cdiv(BLOCK_SIZE_M, tl.warp_size())
    num_warp_n = tl.cdiv(BLOCK_SIZE_N, tl.warp_size())
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    warp_id = tl.program_id(axis=1) % num_pid_in_warp
    warp_m = warp_id // num_warp_n
    warp_n = warp_id % num_warp_n
    lane_id = tl.program_id(axis=2) % tl.warp_size()

    # Pointers for matrix blocks
    a_block_ptr = a_ptr + (pid_m * BLOCK_SIZE_M * stride_am) + (warp_m * tl.warp_size() * stride_am) + (lane_id * stride_am)
    b_block_ptr = b_ptr + (pid_n * BLOCK_SIZE_N * stride_bn) + (warp_n * tl.warp_size() * stride_bn) + (lane_id * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        a_block = tl.load(a_block_ptr + k * stride_ak, mask=k + lane_id < K)
        b_block = tl.load(b_block_ptr + k * stride_bk, mask=k + lane_id < K)
        accumulator += tl.dot(a_block, b_block)

    # Apply leaky ReLU if specified
    if apply_leaky_relu:
        accumulator = tl.where(accumulator > 0, accumulator, alpha * accumulator)

    # Store the result
    c_block_ptr = c_ptr + (pid_m * BLOCK_SIZE_M * stride_cm) + (pid_n * BLOCK_SIZE_N * stride_cn)
    tl.store(c_block_ptr, accumulator, mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < M and tl.arange(0, BLOCK_SIZE_N) < N)

# Define the wrapper function
@triton.jit
def matmul(a, b, apply_leaky_relu=False, alpha=0.01):
    # Validate input tensor compatibility
    M, K = a.shape
    K, N = b.shape
    assert K == b.shape[0], "Matrix dimensions must be compatible for multiplication"

    # Ensure contiguity
    a = a.contiguous()
    b = b.contiguous()

    # Create output tensor
    c = tl.zeros((M, N), dtype=tl.float32)

    # Compute the execution grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), 1, 1)

    # Call the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        apply_leaky_relu=apply_leaky_relu,
        alpha=alpha
    )

    return c

# Define the leaky ReLU function
@triton.jit
def leaky_relu(x, alpha=0.01):
    return tl.where(x > 0, x, alpha * x)
