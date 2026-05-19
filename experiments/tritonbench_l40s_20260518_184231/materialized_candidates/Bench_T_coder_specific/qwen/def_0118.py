import torch
import triton
import triton.language as tl

# Triton kernel signature
@triton.jit
def solve_triangular_kernel(
    A_ptr: tl.tensor, 
    b_ptr: tl.tensor, 
    x_ptr: tl.tensor, 
    n: tl.int32, 
    upper: tl.bool):
    
    # Calculate row and column indices
    row = tl.program_id(0)
    col = tl.program_id(1)

    if row >= n or col >= n:
        return

    # Initialize x[row]
    x = b_ptr[row]

    # Perform forward substitution for upper triangular matrix
    for j in range(row):
        x -= A_ptr[row * n + j] * x_ptr[j]

    # Normalize by diagonal element
    if upper:
        x /= A_ptr[row * n + row]

    # Store result back to global memory
    x_ptr[row] = x

# Wrapper function
def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"
    assert b.ndim in [1, 2] and b.shape[-1] == A.shape[0], "b must have the same number of rows as A"
    assert y.ndim == 1 and y.shape[0] == A.shape[0], "y must have the same number of elements as A"
    
    n = A.shape[0]
    dtype = A.dtype
    
    # Allocate device memory
    A_device = A.contiguous().to(device='cuda', dtype=dtype)
    b_device = b.contiguous().to(device='cuda', dtype=dtype)
    x_device = torch.zeros_like(b, device='cuda', dtype=dtype)
    
    # Launch Triton kernel
    block_size = (16, 16)
    grid_size = ((n + block_size[0] - 1) // block_size[0], (n + block_size[1] - 1) // block_size[1])
    solve_triangular_kernel[grid_size, block_size](A_device, b_device, x_device, n, True)
    
    # Scale and add y
    x_device += alpha * y.to(dtype=dtype).contiguous().to(device='cuda')
    
    return x_device.cpu()
