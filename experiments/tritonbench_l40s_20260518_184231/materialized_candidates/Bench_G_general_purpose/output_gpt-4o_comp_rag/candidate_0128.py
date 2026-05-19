import triton
import triton.language as tl

@triton.jit
def matmul_kernel(C, A, B, M, N, K,
                  stride_cm, stride_cn,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  ACTIVATION: tl.constexpr):
    # Get program IDs for block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate offsets for A and B matrices
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the blocks of A and B
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator for the block of C
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks of A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Perform dot product and accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply optional leaky ReLU activation
    if ACTIVATION == 'leaky_relu':
        accumulator = tl.where(accumulator > 0, accumulator, 0.01 * accumulator)

    # Calculate offsets for C matrix
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # Store the computed block to C
    tl.store(c_ptrs, accumulator)

def matmul(C, A, B, M, N, K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, activation=None):
    # Ensure the activation function is valid
    if activation not in [None, 'leaky_relu']:
        raise ValueError("Invalid activation function. Supported: None, 'leaky_relu'.")

    # Set the activation parameter
    ACTIVATION = activation

    # Calculate grid dimensions
    grid = (tl.cdiv(M, BLOCK_SIZE_M), tl.cdiv(N, BLOCK_SIZE_N))

    # Launch the kernel
    matmul_kernel[grid](
        C, A, B, M, N, K,
        C.stride(0), C.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        ACTIVATION
    )
