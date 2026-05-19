import triton
import triton.language as tl
import numpy as np

# Assuming we have a custom SVD function available
from some_external_library import svd

@triton.jit
def low_rank_svd_approximation(A, k, full_matrices=True, out=None):
    # Get the shapes of the input tensor
    batch_shape = A.shape[:-2]
    m, n = A.shape[-2:]

    # Ensure k is within the valid range
    if not (1 <= k <= min(m, n)):
        raise ValueError(f"k must satisfy 1 <= k <= min(m, n). Got k={k}, m={m}, n={n}")

    # Allocate memory for the output tensor
    if out is None:
        out = tl.zeros(batch_shape + (m, n), dtype=A.dtype)

    # Perform SVD on each batch element
    for i in range(len(batch_shape)):
        for j in range(m):
            for l in range(n):
                u, s, vh = svd(A[i, j, l].contiguous())
                
                # Truncate to rank k
                if not full_matrices:
                    s = s[:k]
                    u = u[:, :k]
                    vh = vh[:k, :]
                
                # Reconstruct the low-rank approximation
                ak = u @ (s[:, None] * vh)
                
                # Store the result in the output tensor
                out[i, j, l] = ak

    return out
