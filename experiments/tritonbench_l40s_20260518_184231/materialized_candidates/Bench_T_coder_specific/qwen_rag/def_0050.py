import triton
import triton.language as tl
import torch
import math

# Kernel function to compute the square root and exponential of elements in a tensor
@triton.jit
def sqrt_exp_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute the square root of the loaded elements
    sqrt_value = tl.sqrt(a_value.to(tl.float32))
    # Apply the exponential function to the square-rooted values
    b_value = tl.exp(sqrt_value)
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b + offset, b_value, mask=mask)

# Wrapper function to launch the Triton kernel and compute sqrt followed by exp
def sqrt_exp(input, out=None) -> torch.Tensor:
    # Create an output tensor 'B' with the same shape as 'input' if 'out' is not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        assert input.shape == out.shape, "Input and output tensors must have the same shape"
    
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    sqrt_exp_kernel[(grid_size, 1, 1)](input, out, n_elements, block_size)
    return out

# Example usage
if __name__ == "__main__":
    input_tensor = torch.tensor([4.0, 9.0, 16.0], dtype=torch.float32)
    output_tensor = sqrt_exp(input_tensor)
    print("Input:", input_tensor)
    print("Output:", output_tensor)
