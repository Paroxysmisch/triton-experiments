import triton
import triton.language as tl
import torch

# Define the leaky ReLU activation function
def leaky_relu(x, negative_slope=0.01):
    return tl.where(x >= 0, x, negative_slope * x)

# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  apply_activation: tl.constexpr, activation_func: tl.constexpr):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define the start of each block in the output matrix C
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Create pointers for each block
    c_ptrs = C + block_start_m * stride_cm + block_start_n * stride_cn

    # Initialize accumulation buffer
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute offsets for A and B
        a_ptrs = A + block_start_m * stride_am + k * stride_ak
        b_ptrs = B + k * stride_bk + block_start_n * stride_bn

        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=(block_start_m + tl.arange(0, BLOCK_SIZE_M)[:, None]) < M)
        b = tl.load(b_ptrs, mask=(block_start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) < N)

        # Compute the matrix multiplication for this block
        acc += tl.dot(a, b)

    # Apply activation function if specified
    if apply_activation:
        acc = activation_func(acc)

    # Write back the result to C
    mask = (block_start_m + tl.arange(0, BLOCK_SIZE_M)[:, None]) < M
    mask = mask & ((block_start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) < N)
    tl.store(c_ptrs, acc, mask=mask)

# Wrapper function for matrix multiplication
def matmul(A, B, activation=None):
    # Get dimensions
    M, K = A.shape
    K, N = B.shape

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Initialize output matrix C
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Define strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Configure grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Determine the activation function
    apply_activation = activation is not None
    activation_func = None
    if activation == 'leaky_relu':
        activation_func = leaky_relu

    # Launch the kernel
    matmul_kernel[grid](
        A, B, C, M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        apply_activation=apply_activation, activation_func=activation_func
    )

    return C

# Example usage
A = torch.randn(256, 256, device='cuda', dtype=torch.float32)
B = torch.randn(256, 256, device='cuda', dtype=torch.float32)
C = matmul(A, B, activation='leaky_relu')
