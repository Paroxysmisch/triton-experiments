import triton
import triton.language as tl

# Define the kernel
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ALPHA, apply_activation, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn):
    # Define the ranges for the block indices
    pid = tl.program_id(axis=0)
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Define the start positions for the blocks
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create pointers to the blocks in A and B
    A_block_ptr = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_block_ptr = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Initialize the accumulator for C
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_SIZE_K
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the blocks of A and B
        A_block = tl.load(A_block_ptr, mask=offs_am[:, None] < M)
        B_block = tl.load(B_block_ptr, mask=offs_bn[None, :] < N)

        # Compute the matrix multiplication for the block
        acc += tl.dot(A_block, B_block)

        # Update the pointers for the next chunk
        A_block_ptr += BLOCK_SIZE_K * stride_ak
        B_block_ptr += BLOCK_SIZE_K * stride_bk

    # Apply optional leaky ReLU activation
    if apply_activation:
        acc = tl.where(acc > 0, acc, acc * ALPHA)

    # Define the output pointer for C
    C_block_ptr = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn

    # Store the result in C
    tl.store(C_block_ptr, acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

# Define the wrapper function
def matmul(A, B, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, ALPHA=0.01, apply_activation=False):
    M, K = A.shape
    _, N = B.shape

    # Ensure A is contiguous
    assert A.is_contiguous()

    # Allocate output matrix C
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        ALPHA, apply_activation,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )

    return C
