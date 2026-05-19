import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr, row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID for parallelization across rows
    row_idx = tl.program_id(0)

    # Calculate the start of the row
    row_start = row_idx * row_stride

    # Create a range of indices for the row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)

    # Load the row data
    row_data = tl.load(input_ptr + offsets, mask=offsets < n_cols, other=-float('inf'))

    # Compute the maximum value for numerical stability
    row_max = tl.max(row_data, axis=0)

    # Subtract the max and exponentiate
    row_data = tl.exp(row_data - row_max)

    # Compute the sum of the exponentiated values
    row_sum = tl.sum(row_data, axis=0)

    # Normalize to get softmax probabilities
    softmax_output = row_data / row_sum

    # Store the result
    tl.store(output_ptr + offsets, softmax_output, mask=offsets < n_cols)

def softmax(input_tensor):
    # Get the dimensions of the input tensor
    n_rows, n_cols = input_tensor.shape

    # Define the block size
    BLOCK_SIZE = 1024  # Adjust based on your GPU's capability

    # Allocate output tensor
    output_tensor = torch.empty_like(input_tensor)

    # Define the grid configuration
    grid = (n_rows,)

    # Launch the kernel
    softmax_kernel[grid](
        input_tensor,
        output_tensor,
        input_tensor.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output_tensor
