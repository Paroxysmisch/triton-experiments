import triton
import triton.language as tl

@triton.jit
def nested3(in_ptr, out_ptr, stride_m, stride_n, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # Compute block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute starting indices for the block
    m_start = pid_m * BLOCK_SIZE
    n_start = pid_n * BLOCK_SIZE
    
    # Loop over the 2x2 tile
    for i in range(2):
        for j in range(2):
            # Calculate row and column indices
            row_idx = m_start + i
            col_idx = n_start + j
            
            # Ensure indices are within bounds
            if row_idx < n_rows and col_idx < n_cols:
                # Calculate pointers
                a_ptrs = in_ptr + row_idx * stride_m + col_idx
                c_ptrs = out_ptr + row_idx * stride_m + col_idx
                
                # Load from input
                a = tl.load(a_ptrs)
                
                # Store to output
                tl.store(c_ptrs, a)

import torch

def wrapper_nested3(n_rows, n_cols):
    # Define block size
    BLOCK_SIZE = 2  # For 2x2 tiles
    
    # Create input and output tensors on CUDA
    x = torch.arange(n_rows * n_cols, dtype=torch.float32, device='cuda').reshape(n_rows, n_cols)
    output = torch.empty_like(x)
    
    # Calculate strides
    stride_m = x.stride(0)
    stride_n = x.stride(1)
    
    # Define grid size
    grid = (n_rows // BLOCK_SIZE, n_cols // BLOCK_SIZE)
    
    # Launch the Triton kernel
    nested3[grid](x, output, stride_m, stride_n, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    # Print the output tensor
    print(output)

# Example usage
n_rows = 4
n_cols = 4
wrapper_nested3(n_rows, n_cols)
