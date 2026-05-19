import triton
import triton.language as tl

@triton.jit
def nested3(in_ptr, out_ptr, n_rows, n_cols, stride_m, stride_n, BLOCK_SIZE: tl.constexpr):
    # Compute the tile coordinates
    pid = tl.program_id(0)
    num_tiles = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    tile_id = pid % num_tiles
    tile_start = tile_id * BLOCK_SIZE

    # Iterate over the 2x2 tile
    for i in range(2):
        for j in range(2):
            for k in range(BLOCK_SIZE):
                # Calculate the current indices
                row = i * 2 + j
                col = tile_start + k

                # Calculate the pointers
                a_ptrs = in_ptr + row * stride_m + col * stride_n
                c_ptrs = out_ptr + row * stride_m + col * stride_n

                # Load and store
                a_val = tl.load(a_ptrs)
                tl.store(c_ptrs, a_val)

import torch

def wrapper_nested3(n_rows, n_cols):
    # Define the block size
    BLOCK_SIZE = 4

    # Create input and output tensors
    x = torch.arange(n_rows * n_cols, dtype=torch.float32, device='cuda').reshape(n_rows, n_cols)
    output = torch.empty_like(x)

    # Define the grid configuration
    grid = (n_cols // BLOCK_SIZE,)

    # Define the strides
    stride_m = x.stride(0)
    stride_n = x.stride(1)

    # Launch the kernel
    nested3[grid](x, output, n_rows, n_cols, stride_m, stride_n, BLOCK_SIZE)

    # Print the output tensor
    print(output)
