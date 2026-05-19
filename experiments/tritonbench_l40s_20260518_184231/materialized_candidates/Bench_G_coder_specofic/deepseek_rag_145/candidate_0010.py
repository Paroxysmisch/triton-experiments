import triton.language as tl
import torch

@triton.jit
def mv_kernel(A_ptr, B_ptr, C_ptr, N, M, BLOCK_N, BLOCK_M):
    n = tl.program_id(0)
    m = tl.program_id(1)

    # Allocate shared memory for the sub-matrix
    sA = tl.shared.array((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
    # Load the sub-matrix from A and B to shared memory
    sA[m, n] = A_ptr[n, m]
    B = B_ptr[m]

    tl.barrier()

    # Compute the partial sum
    partial_sum = 0.0
    for i in range(BLOCK_M):
        partial_sum += sA[i, n] * B[i]

    # Write the result to C
    if n == 0:
        C_ptr[m] = partial_sum

def mv(A, B, C):
    # Assert dimensions compatibility
    assert A.shape[0] == C.shape[0]
    assert A.shape[1] == B.shape[0]
    assert B.shape[1] == C.shape[1]

    # Manage CUDA resources for launching the kernel with calculated grid dimensions
    grid = lambda M, N: ((N + M - 1) // M, (M + N - 1) // N)
    dtype = torch.float32
    A_torch = A.to('cuda')
    B_torch = B.to('cuda')
    C_torch = C.to('cuda')

    A_ptr = triton.pointers.numpy(A_torch.flatten(), dtype)
    B_ptr = triton.pointers.numpy(B_torch.flatten(), dtype)
    C_ptr = triton.pointers.numpy(C_torch.flatten(), dtype)

    N, M = A.shape[0], A.shape[1]

    # Execute the kernel
    mv_kernel[grid(M, N)](A_ptr, B_ptr, C_ptr, N, M, M, N)
