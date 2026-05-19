import triton
import triton.language as tl

@triton.jit
def eig_kernel(A_ptr, V_ptr, Lambda_ptr, n, BATCH_SIZE):
    # Get the grid and thread ids
    block_id = tl.program_id(0)
    thread_id = tl.program_id(1)

    # Compute the batch id
    batch_id = block_id // n

    # Compute the offsets
    A_offset = batch_id * BATCH_SIZE * n * n + thread_id
    V_offset = batch_id * BATCH_SIZE * n * n + thread_id
    Lambda_offset = batch_id * BATCH_SIZE * n + thread_id

    # Load the input matrix A
    A = tl.load(A_ptr + A_offset)

    # Perform the eigenvalue decomposition
    V, Lambda = tl.linalg.eig(A)

    # Store the eigenvectors and eigenvalues
    tl.store(V_ptr + V_offset, V)
    tl.store(Lambda_ptr + Lambda_offset, Lambda)

def eig(A, out=None):
    # Get the shape of the input tensor
    n = A.shape[-1]
    BATCH_SIZE = A.numel() // n**2

    # Allocate memory for the output tensors
    V = triton.empty((BATCH_SIZE, n, n), dtype=A.dtype)
    Lambda = triton.empty((BATCH_SIZE, n), dtype=A.dtype)

    # Launch the kernel
    eig_kernel[BATCH_SIZE, n](A.data, V.data, Lambda.data, n, BATCH_SIZE)

    return V, Lambda
