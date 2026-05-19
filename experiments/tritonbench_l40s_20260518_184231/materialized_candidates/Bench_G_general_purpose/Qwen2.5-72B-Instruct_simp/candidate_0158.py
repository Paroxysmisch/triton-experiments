import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr, M, N, B, BLOCK_N_SIZE: tl.constexpr):
    # Compute the block ID in the M and N dimensions
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    # Compute the block range in the M and N dimensions
    block_start_m = pid_m * BLOCK_N_SIZE
    block_start_n = pid_n * BLOCK_N_SIZE

    # Compute the block range in the batch dimension
    batch_start = pid_b * M * N

    # Initialize the output block
    output_block = tl.zeros((BLOCK_N_SIZE, BLOCK_N_SIZE), dtype=tl.float32)

    # Load the weight vector
    rms_w = tl.load(rms_w_ptr + block_start_n)

    # Iterate over the elements in the block
    for i in range(BLOCK_N_SIZE):
        # Compute the start and end indices for the current row
        row_start = block_start_m + i * N
        row_end = row_start + BLOCK_N_SIZE

        # Load the row from the input tensor
        row = tl.load(x_ptr + batch_start + row_start + block_start_n)

        # Compute the mean of the squared values
        mean = tl.sum(row * row, axis=0) / BLOCK_N_SIZE

        # Compute the root mean square
        rms = tl.sqrt(mean + 1e-6)

        # Normalize the row
        normalized_row = row / rms

        # Scale the normalized row by the weight vector
        scaled_row = normalized_row * rms_w

        # Store the result in the output block
        output_block[i, :] = scaled_row

    # Store the output block back to the output tensor
    tl.store(output_ptr + batch_start + block_start_m * N + block_start_n, output_block)

import torch

def rmsnorm_triton_wrapper(x, rms_w, output):
    # Get the dimensions of the input tensor
    B, M, N = x.shape

    # Define the block size
    BLOCK_N_SIZE = 128

    # Compute the number of blocks in the M and N dimensions
    grid_m = (M + BLOCK_N_SIZE - 1) // BLOCK_N_SIZE
    grid_n = (N + BLOCK_N_SIZE - 1) // BLOCK_N_SIZE

    # Define the grid and block dimensions
    grid = (grid_m, grid_n, B)

    # Launch the kernel
    rmsnorm_triton[grid, (BLOCK_N_SIZE, BLOCK_N_SIZE)](
        x, rms_w, output, M, N, B, BLOCK_N_SIZE
    )

# Example usage
B, M, N = 2, 1024, 1024
x = torch.randn((B, M, N), device='cuda')
rms_w = torch.randn((N,), device='cuda')
output = torch.empty((B, M, N), device='cuda')

rmsnorm_triton_wrapper(x, rms_w, output)

print(output)
