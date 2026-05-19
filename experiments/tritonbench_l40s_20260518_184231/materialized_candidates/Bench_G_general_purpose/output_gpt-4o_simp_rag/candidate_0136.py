import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    dtype: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # Compute the row and column indices for the current block
    block_row = pid // (N // BLOCK_N)
    block_col = pid % (N // BLOCK_N)

    # Pointers to the block of A and B
    a_block_ptr = a_ptr + block_row * BLOCK_M * K
    b_block_ptr = b_ptr + block_col * BLOCK_N

    # Initialize an accumulator for the block of C
    c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in steps of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Load blocks of A and B into shared memory
        a_block = tl.load(a_block_ptr + k * BLOCK_M)
        b_block = tl.load(b_block_ptr + k * BLOCK_N * K)

        # Compute matrix multiplication for the block
        c_block += tl.dot(a_block, b_block)

    # Write the result back to the output matrix C
    c_ptr = c_ptr + block_row * BLOCK_M * N + block_col * BLOCK_N
    tl.store(c_ptr, c_block)

def matmul_persistent(a, b, dtype=torch.float16):
    # Ensure input dimensions are compatible
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"

    # Define grid and block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    M, K = a.shape
    _, N = b.shape

    # Allocate output tensor
    c = torch.empty((M, N), dtype=dtype, device=a.device)

    # Define grid size
    grid = (M // BLOCK_M) * (N // BLOCK_N)

    # Launch the kernel
    matmul_kernel_persistent[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        dtype=tl.float16 if dtype == torch.float16 else tl.float32
    )

    return c

# Example usage
a = torch.randn((1024, 512), dtype=torch.float16, device='cuda')
b = torch.randn((512, 1024), dtype=torch.float16, device='cuda')
c = matmul_persistent(a, b)
