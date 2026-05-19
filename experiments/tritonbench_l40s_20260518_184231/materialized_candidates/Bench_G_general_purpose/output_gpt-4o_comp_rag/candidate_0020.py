import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Identify the current row
    row_idx = tl.program_id(0)
    # Calculate the starting pointer for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # Define column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into on-chip memory with masking
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_f32 = row.to(tl.float32)
    # Perform numerical stabilization
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)
    # Compute exponentials
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Store the result back in the output matrix
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)

def softmax(input_matrix):
    # Determine the matrix dimensions
    n_rows, n_cols = input_matrix.shape
    # Calculate the optimal BLOCK_SIZE (smallest power of two greater than n_cols)
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2
    # Adjust num_warps according to BLOCK_SIZE
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Initialize the output matrix
    output_matrix = torch.empty_like(input_matrix)
    # Launch the Triton kernel with one block per row
    grid = (n_rows,)
    softmax_kernel[grid](output_matrix, input_matrix, input_matrix.stride(0), output_matrix.stride(0), n_cols, BLOCK_SIZE, num_warps=num_warps)
    return output_matrix
