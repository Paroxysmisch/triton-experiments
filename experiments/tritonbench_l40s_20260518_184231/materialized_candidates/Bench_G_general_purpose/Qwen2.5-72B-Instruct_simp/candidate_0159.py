import triton
import triton.language as tl

@triton.jit
def nested3(in_ptr, out_ptr, stride_n, M, N, BLOCK_SIZE: tl.constexpr):
    # Compute the position of the block in the grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the block boundaries
    rm = pid_m * BLOCK_SIZE
    rn = pid_n * BLOCK_SIZE
    
    # Compute the block boundaries
    bounds_m = tl.min(rm + BLOCK_SIZE, M)
    bounds_n = tl.min(rn + BLOCK_SIZE, N)
    
    # Load the input block
    in_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if rm + i < M and rn + j < N:
                in_block[i, j] = tl.load(in_ptr + (rm + i) * stride_n + (rn + j))
    
    # Shift the block and store it to the output
    out_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if rm + i < M and rn + j < N:
                out_block[i, j] = in_block[i, j]
    
    # Store the output block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if rm + i < M and rn + j < N:
                tl.store(out_ptr + (rm + i) * stride_n + (rn + j), out_block[i, j])

import torch
import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 2

# Define the kernel
@triton.jit
def nested3(in_ptr, out_ptr, stride_n, M, N, BLOCK_SIZE: tl.constexpr):
    # Compute the position of the block in the grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the block boundaries
    rm = pid_m * BLOCK_SIZE
    rn = pid_n * BLOCK_SIZE
    
    # Compute the block boundaries
    bounds_m = tl.min(rm + BLOCK_SIZE, M)
    bounds_n = tl.min(rn + BLOCK_SIZE, N)
    
    # Load the input block
    in_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if rm + i < M and rn + j < N:
                in_block[i, j] = tl.load(in_ptr + (rm + i) * stride_n + (rn + j))
    
    # Shift the block and store it to the output
    out_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if rm + i < M and rn + j < N:
                out_block[i, j] = in_block[i, j]
    
    # Store the output block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if rm + i < M and rn + j < N:
                tl.store(out_ptr + (rm + i) * stride_n + (rn + j), out_block[i, j])

# Define the wrapper function
def wrapper_nested3(M, N, stride_n):
    # Initialize the input and output tensors
    in_tensor = torch.arange(M * N, dtype=torch.float32).view(M, N).cuda()
    out_tensor = torch.zeros_like(in_tensor)
    
    # Define the grid and block dimensions
    grid = (M // BLOCK_SIZE, N // BLOCK_SIZE)
    
    # Launch the kernel
    nested3[grid](in_tensor, out_tensor, stride_n, M, N, BLOCK_SIZE)
    
    # Print the output tensor
    print("Output Tensor:")
    print(out_tensor.cpu().numpy())

# Example usage
M = 4
N = 4
stride_n = N
wrapper_nested3(M, N, stride_n)
