import triton
import triton.language as tl

@triton.jit
def cholesky_solve_kernel(B_ptr, L_ptr, out_ptr, n, k, upper, n_batch):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Load the right-hand side tensor B
    B = tl.load(B_ptr + batch_idx * n * k * tl.numel(B_ptr))
    
    # Initialize output tensor
    X = tl.zeros((n, k), dtype=tl.float32)  # Adjust dtype based on input
    
    # Perform Cholesky solve
    for i in range(n - 1, -1, -1):
        sum = B[i]
        for j in range(i + 1, n):
            sum -= L[i, j] * X[j]
        X[i] = sum / L[i, i]
    
    # Store the result in the output tensor
    tl.store(out_ptr + batch_idx * n * k * tl.numel(out_ptr), X)


def cholesky_solve(B, L, upper=False, *, out=None):
    # Validate input shapes
    if B.ndim < 2 or L.ndim < 2:
        raise ValueError("B and L must have at least 2 dimensions.")
    
    # Get batch dimensions
    n_batch = B.shape[:-2]
    n, k = B.shape[-2], B.shape[-1]
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(B)
    
    # Launch the Triton kernel
    grid = (n_batch[0],)  # Adjust grid size based on batch dimensions
    cholesky_solve_kernel[grid](B, L, out, n, k, upper, n_batch)
    
    return out
