import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def load_reduce_kernel(
    input_ptr,  # Pointer to the input matrix
    output_ptr,  # Pointer to the output vector
    stride_am,  # Stride of the input matrix in the first dimension
    stride_an,  # Stride of the input matrix in the second dimension
    M,  # Number of rows in the input matrix
    N,  # Number of columns in the input matrix
    BLOCK_M: tl.constexpr,  # Block size in the first dimension
    BLOCK_N: tl.constexpr  # Block size in the second dimension
):
    # Program ID for the block in the first dimension
    pid_m = tl.program_id(0)

    # Compute the start index for the block
    block_start_m = pid_m * BLOCK_M

    # Define a mask to handle boundaries
    mask_m = block_start_m + tl.arange(0, BLOCK_M) < M

    # Initialize a max accumulator with a very low value
    max_vals = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)

    # Iterate over the second dimension in blocks
    for block_start_n in range(0, N, BLOCK_N):
        # Load a block of data from the input matrix
        offs_n = block_start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Compute the linear indices for the load
        offs = block_start_m[:, None] * stride_am + offs_n[None, :] * stride_an

        # Load the data with mask
        data = tl.load(input_ptr + offs, mask=mask_m[:, None] & mask_n[None, :], other=-float('inf'))

        # Reduce along the second dimension to find the maximum
        max_vals = tl.maximum(max_vals, tl.max(data, axis=1))

    # Store the result in the output vector
    tl.store(output_ptr + block_start_m, max_vals, mask=mask_m)

# Wrapper function to test the kernel
def load_reduce(input_matrix, BLOCK_M=128, BLOCK_N=128):
    # Get the dimensions of the input matrix
    M, N = input_matrix.shape

    # Allocate output vector
    output_vector = torch.empty(M, dtype=input_matrix.dtype, device=input_matrix.device)

    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_M),)
    load_reduce_kernel[grid](
        input_ptr=input_matrix,
        output_ptr=output_vector,
        stride_am=input_matrix.stride(0),
        stride_an=input_matrix.stride(1),
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )

    return output_vector

# Example usage
if __name__ == "__main__":
    # Create a random input matrix
    M, N = 1024, 512
    input_matrix = torch.randn((M, N), device='cuda')

    # Compute the maximum along the second dimension
    output_vector = load_reduce(input_matrix)

    # Validate the result
    expected_output = input_matrix.max(dim=1)[0]
    assert torch.allclose(output_vector, expected_output), "Mismatch between Triton and PyTorch results!"
    print("Test passed!")
