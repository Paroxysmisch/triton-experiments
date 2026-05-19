import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols, BLOCK_SIZE: tl.constexpr
):
    # Row index of the current program.
    row_idx = tl.program_id(0)

    # Compute the start of the row in the input and output matrices.
    input_offset = row_idx * input_row_stride
    output_offset = row_idx * output_row_stride

    # Create a pointer for the start of the row.
    input_ptr_row = input_ptr + input_offset
    output_ptr_row = output_ptr + output_offset

    # Load data for the current row.
    row_data = tl.load(input_ptr_row + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))

    # Numerical stability: subtract max from row elements.
    row_max = tl.max(row_data, axis=0)
    row_data = row_data - row_max

    # Exponentiate the stabilized values.
    exp_row_data = tl.exp(row_data)

    # Sum of exponentials for normalization.
    sum_exp_row_data = tl.sum(exp_row_data, axis=0)

    # Normalize to get softmax.
    softmax_row_data = exp_row_data / sum_exp_row_data

    # Store the result in the output matrix.
    tl.store(output_ptr_row + tl.arange(0, BLOCK_SIZE), softmax_row_data, mask=tl.arange(0, BLOCK_SIZE) < n_cols)


def softmax(input_tensor):
    # Get the dimensions of the input matrix.
    n_rows, n_cols = input_tensor.shape

    # Determine BLOCK_SIZE and number of warps.
    BLOCK_SIZE = 128  # This can be tuned based on the hardware.
    num_warps = 4     # Also tunable.

    # Allocate output tensor.
    output_tensor = torch.empty_like(input_tensor)

    # Launch the kernel with a 1D grid where each block handles one row.
    grid = (n_rows,)

    # Call the Triton kernel.
    softmax_kernel[grid](
        output_tensor, input_tensor,
        input_tensor.stride(0), output_tensor.stride(0),
        n_cols, BLOCK_SIZE=BLOCK_SIZE
    )

    return output_tensor

# Example usage:
if __name__ == "__main__":
    # Create a random input tensor.
    input_tensor = torch.randn(1024, 512, device='cuda', dtype=torch.float32)

    # Compute softmax using the Triton-based implementation.
    output_tensor = softmax(input_tensor)

    # Verify the result using PyTorch's softmax.
    expected_output = torch.nn.functional.softmax(input_tensor, dim=1)

    # Check if the results are close.
    assert torch.allclose(output_tensor, expected_output, atol=1e-5)
