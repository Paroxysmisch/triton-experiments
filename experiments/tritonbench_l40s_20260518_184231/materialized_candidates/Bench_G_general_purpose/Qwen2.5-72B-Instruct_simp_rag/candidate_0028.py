import triton
import triton.language as tl

# Define the softmax kernel
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_f32 = row.to(tl.float32)
    # Subtract maximum for numerical stability
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)

# Define the softmax wrapper function
def softmax(input_matrix, output_matrix, n_rows, n_cols, BLOCK_SIZE=1024, num_warps=4):
    # Determine the number of warps based on the block size
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # Set up the grid and block dimensions
    grid = (n_rows,)

    # Launch the kernel
    softmax_kernel[grid](output_matrix, input_matrix, n_cols, n_cols, n_cols, BLOCK_SIZE, num_warps=num_warps)

# Example usage
if __name__ == "__main__":
    import torch

    # Example input matrix
    n_rows, n_cols = 1024, 1024
    input_matrix = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.float32)
    output_matrix = torch.empty_like(input_matrix)

    # Compute softmax using the Triton kernel
    softmax(input_matrix, output_matrix, n_rows, n_cols, BLOCK_SIZE=1024, num_warps=4)

    # Print the first row of the output matrix
    print(output_matrix[0])
