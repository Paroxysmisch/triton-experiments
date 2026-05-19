{{ code }}
import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the index of the current element
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements  # Ensure we don't go out of bounds

    # Load input tensor
    input_data = tl.load(input_ptr + idx, mask=mask)

    # Apply ReLU operation
    output_data = tl.maximum(input_data, 0)

    # Store the result back to output tensor
    tl.store(output_ptr + idx, output_data, mask=mask)

def relu(input, inplace=False):
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    
    # Create output tensor
    output = input if inplace else input.clone()

    # Define the block size for the kernel
    BLOCK_SIZE = 1024  # You can adjust this based on your GPU architecture

    # Launch the kernel
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    relu_kernel[grid](input.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE)

    return output
{{ code }}
