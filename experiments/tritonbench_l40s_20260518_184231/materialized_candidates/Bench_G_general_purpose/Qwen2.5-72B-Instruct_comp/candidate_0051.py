import triton
import triton.language as tl

# Constants
BLOCK_SIZE = 1024

# Kernel 1: Compute max values within blocks
@triton.jit
def max_kernel_1(input_ptr, mid_ptr, n_elements, pid, BLOCK_SIZE: tl.constexpr):
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the data with masking
    x = tl.load(input_ptr + offsets, mask=mask)

    # Compute the max value in this block
    block_max = tl.max(x, axis=0)

    # Store the max value in the intermediate tensor
    tl.store(mid_ptr + pid, block_max)

# Kernel 2: Consolidate results from max_kernel_1
@triton.jit
def max_kernel_2(mid_ptr, out_ptr, n_blocks, pid, BLOCK_SIZE: tl.constexpr):
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks

    # Load the data with masking
    x = tl.load(mid_ptr + offsets, mask=mask)

    # Compute the max value in this block
    block_max = tl.max(x, axis=0)

    # Store the final max value in the output tensor
    if pid == 0:
        tl.store(out_ptr, block_max)

# Advanced Kernel: Multi-dimensional max computation
@triton.jit
def max_kernel(input_ptr, out_ptr, M, N, K, pid_m, pid_k, dim, BLOCK_SIZE: tl.constexpr):
    # Compute the starting index for this block
    row = pid_m
    col = pid_k

    # Compute the offset in the input tensor
    offsets = row * K + col
    mask = offsets < M * K

    # Load the data with masking
    x = tl.load(input_ptr + offsets, mask=mask)

    # Compute the max value and index
    max_val = tl.max(x, axis=0)
    max_idx = tl.argmax(x, axis=0)

    # Store the max value and index in the output tensor
    if dim == 0:
        tl.store(out_ptr + col, max_val)
    else:
        tl.store(out_ptr + row, max_val)

# Wrapper function for max computation
def max(input_tensor, out_tensor, BLOCK_SIZE=BLOCK_SIZE):
    n_elements = input_tensor.shape[0]
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create intermediate tensor
    mid_tensor = triton.empty((n_blocks,), dtype=input_tensor.dtype, device=input_tensor.device)

    # Execute max_kernel_1
    max_kernel_1[(n_blocks,)](input_tensor, mid_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Execute max_kernel_2
    max_kernel_2[(1,)](mid_tensor, out_tensor, n_blocks, BLOCK_SIZE=BLOCK_SIZE)

# Wrapper function for multi-dimensional max computation
def max_dim(input_tensor, out_tensor, dim, BLOCK_SIZE=BLOCK_SIZE):
    rank = len(input_tensor.shape)
    if dim < 0 or dim >= rank:
        raise ValueError("Invalid dimension")

    # Reshape the input tensor
    M = input_tensor.shape[dim]
    N = 1
    K = 1
    for i in range(rank):
        if i < dim:
            N *= input_tensor.shape[i]
        elif i > dim:
            K *= input_tensor.shape[i]

    # Compute grid dimensions
    grid_m = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Execute max_kernel
    max_kernel[(grid_m, grid_k)](input_tensor, out_tensor, N, M, K, BLOCK_SIZE=BLOCK_SIZE)

# Example usage
import torch

# Create a large 1D tensor
input_tensor = torch.randn(1000000, device='cuda')
out_tensor = torch.empty(1, device='cuda')

# Compute the max value
max(input_tensor, out_tensor)

# Print the result
print("Max value:", out_tensor.item())

# Create a multi-dimensional tensor
input_tensor_2d = torch.randn(1000, 1000, device='cuda')
out_tensor_2d = torch.empty(1000, device='cuda')

# Compute the max value along dimension 1
max_dim(input_tensor_2d, out_tensor_2d, dim=1)

# Print the result
print("Max values along dimension 1:", out_tensor_2d)
