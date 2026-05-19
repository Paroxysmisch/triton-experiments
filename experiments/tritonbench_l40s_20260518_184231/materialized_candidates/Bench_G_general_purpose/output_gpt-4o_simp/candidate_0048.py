import triton
import triton.language as tl

# Kernel to compute the mean along specified dimensions
@triton.jit
def mean_dim_kernel(X_ptr, mean_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Program ID for distributed computation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current block
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask to handle out-of-bounds access
    mask_m = offsets_m < M
    mask_n = offsets_n < N

    # Load the block of data
    X_block = tl.load(X_ptr + offsets_m[:, None] * N + offsets_n[None, :], mask=mask_m[:, None] & mask_n[None, :])

    # Compute the mean along the N dimension
    mean_value = tl.sum(X_block, axis=1) / N

    # Store the result
    tl.store(mean_ptr + offsets_m, mean_value, mask=mask_m)

# Function to compute mean along specified dimension
def mean_dim(X, dim):
    assert X.ndim == 2, "Input tensor must be 2D"
    
    # Reorder dimensions if needed
    if dim == 1:
        X = X.T

    M, N = X.shape
    BLOCK_M = 128  # Block size for M dimension
    BLOCK_N = 128  # Block size for N dimension

    # Output tensor for mean values
    mean = torch.empty(M, dtype=X.dtype, device=X.device)

    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mean_dim_kernel[grid](X, mean, M, N, BLOCK_M, BLOCK_N)

    return mean

# Example usage
import torch

# Create a 2D tensor
X = torch.randn(256, 256, device='cuda')

# Compute mean along dimension 1
mean_result = mean_dim(X, dim=1)
print(mean_result)
