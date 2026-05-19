import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Get the program ID and compute the starting indices for this block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the start of the block in the output matrix
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Initialize an accumulator for the block
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load a block of A and B into shared memory
        a_block = tl.load(a_ptr + (start_m + tl.arange(0, BLOCK_M))[:, None] * stride_am + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak, mask=(start_m + tl.arange(0, BLOCK_M))[:, None] < M)
        b_block = tl.load(b_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_bn, mask=(start_n + tl.arange(0, BLOCK_N))[None, :] < N)

        # Perform the dot product
        acc += tl.dot(a_block, b_block)

    # Write the result to the output matrix
    c_block_ptr = c_ptr + (start_m + tl.arange(0, BLOCK_M))[:, None] * stride_cm + (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_cn
    tl.store(c_block_ptr, acc, mask=(start_m + tl.arange(0, BLOCK_M))[:, None] < M)

def matmul(a, b):
    # Check input dimensions
    assert a.shape[1] == 4 * b.shape[0], "The number of columns in 'a' must be four times the number of rows in 'b'."

    # Define matrix dimensions
    M, K = a.shape
    K, N = b.shape

    # Create an output matrix
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define the block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Calculate grid sizes
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the kernel
    matmul_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bn=b.stride(1), stride_bk=b.stride(0),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return c

# Example usage:
# a = torch.randn((M, 4*K), device='cuda', dtype=torch.float32)
# b = torch.randn((K, N), device='cuda', dtype=torch.float32)
# c = matmul(a, b)
