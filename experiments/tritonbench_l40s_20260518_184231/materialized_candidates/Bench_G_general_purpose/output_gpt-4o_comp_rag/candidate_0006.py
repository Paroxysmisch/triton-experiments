import triton
import triton.language as tl
import torch

@triton.jit
def mv_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    # Calculate the block index and size
    pid = tl.program_id(axis=0)
    block_start_n = pid * BLOCK_N
    offsets_n = block_start_n + tl.arange(0, BLOCK_N)
    offsets_m = tl.arange(0, BLOCK_M)

    # Load the sub-matrix of A and vector B
    A_block_ptr = A_ptr + offsets_n[:, None] * M + offsets_m[None, :]
    B_block_ptr = B_ptr + offsets_m

    # Initialize accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over blocks of M dimension
    for m_offset in range(0, M, BLOCK_M):
        A_block = tl.load(A_block_ptr + m_offset)
        B_block = tl.load(B_block_ptr + m_offset)
        
        # Element-wise multiplication and accumulation
        acc += tl.dot(A_block, B_block)

    # Write the result to C
    mask = offsets_n < N
    tl.store(C_ptr + offsets_n, acc, mask=mask)

def mv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Check dimensions
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matrix-vector multiplication"
    
    N, M = A.shape
    BLOCK_N = 128  # Example block size, can be tuned
    BLOCK_M = 128  # Example block size, can be tuned

    # Allocate output tensor
    C = torch.empty((N,), device=A.device, dtype=A.dtype)

    # Launch the kernel
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)
    mv_kernel[grid](A, B, C, N, M, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M)

    return C

# Example usage
A = torch.randn(1024, 512, device='cuda')
B = torch.randn(512, device='cuda')
C = mv(A, B)
print(C)
