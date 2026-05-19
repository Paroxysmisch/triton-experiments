import triton
from triton.language import *

@triton.jit
def cholesky_kernel(
    A_ptr: ptr(float32),
    n: int64,
    upper: bool,
    workspace_ptr: ptr(float32),
    result_ptr: ptr(float32),
    batch_size: int64,
    BLOCK_SIZE: int64 = 32):
    
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(batch_size * n, BLOCK_SIZE)

    row = pid % n
    col = pid // n

    # Load A into shared memory
    shared_A = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=float32)
    shared_workspace = tl.zeros(BLOCK_SIZE, dtype=float32)
    row_in_block = row % BLOCK_SIZE
    col_in_block = col % BLOCK_SIZE
    if row_in_block < n and col_in_block < n:
        shared_A[row_in_block, col_in_block] = tl.load(A_ptr + (pid * n + col_in_block))

    # Perform Cholesky decomposition
    for k in range(n):
        if row_in_block == k:
            shared_workspace[k] = tl.sqrt(shared_A[k, k])
            for j in range(k + 1, n):
                shared_A[j, k] /= shared_workspace[k]
        
        # Synchronize across threads in the block
        tl.barrier(tl.scope.block)

        if not upper:
            for i in range(k + 1, n):
                shared_A[i, k] *= shared_workspace[k]
        else:
            for i in range(0, k + 1):
                shared_A[i, k] *= shared_workspace[k]

        # Synchronize across threads in the block
        tl.barrier(tl.scope.block)

    # Store the result back to global memory
    if row_in_block < n and col_in_block < n:
        if upper:
            tl.store(result_ptr + (pid * n + col_in_block), tl.conj(shared_A[col_in_block, row_in_block]))
        else:
            tl.store(result_ptr + (pid * n + col_in_block), shared_A[row_in_block, col_in_block])

@triton.autotune
def cholesky(
    A: tl.tensor,
    *,
    upper: bool = False,
    out: tl.tensor = None,
    BLOCK_SIZE: int64 = 32):
    
    if out is None:
        out = tl.zeros_like(A)

    assert A.ndim >= 2 and A.shape[-2:] == A.shape[-2:], "Input must be a square matrix"
    assert A.dtype in [tl.float32], "Only float32 is supported"

    batch_size = A.shape[:-2].numel()
    n = A.shape[-1]

    cholesky_kernel[
        batch_size * n,
        BLOCK_SIZE
    ](A.data, n, upper, None, out.data, batch_size)

    return out
