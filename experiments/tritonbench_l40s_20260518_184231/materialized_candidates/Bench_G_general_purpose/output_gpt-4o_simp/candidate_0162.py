import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def matmul_kernel(x_ptr, y_ptr, z_ptr, M, N, K, stride_xm, stride_xk, stride_yn, stride_yk, stride_zm, stride_zn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Program ID determines the block of the output matrix
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting indices of the block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over k dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute offsets for x and y
        x_offset = offs_m[:, None] * stride_xm + (offs_k[None, :] + k) * stride_xk
        y_offset = (offs_k[:, None] + k) * stride_yk + offs_n[None, :] * stride_yn

        # Load blocks from x and y
        x = tl.load(x_ptr + x_offset, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        y = tl.load(y_ptr + y_offset, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)

        # Accumulate dot product
        acc += tl.dot(x, y)

    # Compute offset for z
    z_offset = offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn

    # Store the result in z
    tl.store(z_ptr + z_offset, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Wrapper function for the Triton kernel
def matmul(x, y):
    # Get shapes
    M, K = x.shape
    K, N = y.shape

    # Allocate output matrix
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the kernel
    matmul_kernel[grid](
        x, y, z,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return z
