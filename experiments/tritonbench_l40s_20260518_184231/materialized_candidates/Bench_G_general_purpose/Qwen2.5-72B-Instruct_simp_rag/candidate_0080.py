import triton
import triton.language as tl
import torch

@triton.jit
def load_reduce_kernel(
    input_ptr,  # Pointer to the input matrix
    output_ptr,  # Pointer to the output vector
    stride_im,  # Stride of the input matrix along the first dimension
    stride_in,  # Stride of the input matrix along the second dimension
    stride_on,  # Stride of the output vector
    M,  # Number of rows in the input matrix
    N,  # Number of columns in the input matrix
    BLOCK_M: tl.constexpr,  # Block size for the first dimension
    BLOCK_N: tl.constexpr  # Block size for the second dimension
):
    # Compute the block ID in the first dimension
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the block bounds
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    block_end_m = min(block_start_m + BLOCK_M, M)
    block_end_n = min(block_start_n + BLOCK_N, N)

    # Initialize the maximum value for this block
    max_value = -tl.inf

    # Iterate over the block
    for i in range(block_start_m, block_end_m):
        for j in range(block_start_n, block_end_n):
            # Load the value from the input matrix
            value = tl.load(input_ptr + i * stride_im + j * stride_in)
            # Update the maximum value
            max_value = tl.max(max_value, value)

    # Write the maximum value to the output vector
    tl.store(output_ptr + pid_m * stride_on, max_value)

def load_reduce(input: torch.Tensor, output: torch.Tensor, block_m: int, block_n: int):
    # Check input and output dimensions
    assert input.dim() == 2, "Input must be a 2D tensor"
    assert output.dim() == 1, "Output must be a 1D tensor"
    assert output.size(0) == input.size(0), "Output size must match the number of rows in the input"

    # Get the dimensions of the input matrix
    M, N = input.size()

    # Define the grid and block dimensions
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    # Launch the kernel
    load_reduce_kernel[grid](
        input,  # Pointer to the input matrix
        output,  # Pointer to the output vector
        input.stride(0),  # Stride of the input matrix along the first dimension
        input.stride(1),  # Stride of the input matrix along the second dimension
        output.stride(0),  # Stride of the output vector
        M,  # Number of rows in the input matrix
        N,  # Number of columns in the input matrix
        BLOCK_M=block_m,  # Block size for the first dimension
        BLOCK_N=block_n  # Block size for the second dimension
    )

# Test harness
if __name__ == "__main__":
    # Define the block sizes
    BLOCK_M = 16
    BLOCK_N = 16

    # Generate random input data
    M = 128
    N = 128
    input = torch.randn((M, N), device="cuda")

    # Allocate output tensor
    output = torch.empty((M,), device="cuda")

    # Run the kernel
    load_reduce(input, output, BLOCK_M, BLOCK_N)

    # Validate the results
    expected_output = torch.max(input, dim=1).values
    assert torch.allclose(output, expected_output), "Output does not match expected result"
    print("Test passed!")
