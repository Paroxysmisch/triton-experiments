import triton
import triton.language as tl
import torch
import math

# Kernel function to compute the erfc and square root of elements in a tensor
@triton.jit
def erfc_sqrt_kernel(input, erfc_output, sqrt_output, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'input' with boundary mask
    input_value = tl.load(input + offset, mask=mask)
    # Compute the erfc of the loaded elements
    erfc_value = 1.0 - (2.0 / tl.sqrt(tl.float32(math.pi))) * tl.math.erf(input_value)
    # Compute the square root of the loaded elements
    sqrt_value = tl.sqrt(input_value.to(tl.float32))
    # Store the results in the output tensors with boundary mask
    tl.store(erfc_output + offset, erfc_value, mask=mask)
    tl.store(sqrt_output + offset, sqrt_value, mask=mask)

# Wrapper function to launch the Triton kernel and compute erfc and square root
def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create output tensors 'erfc_output' and 'sqrt_output' with the same shape as 'input'
    erfc_output = torch.empty_like(input)
    sqrt_output = torch.empty_like(input)
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    erfc_sqrt_kernel[(grid_size, 1, 1)](input, erfc_output, sqrt_output, n_elements, block_size)
    return erfc_output, sqrt_output

# Example usage
if __name__ == "__main__":
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    erfc_result, sqrt_result = erfc_sqrt(input_tensor)
    print("erfc result:", erfc_result)
    print("sqrt result:", sqrt_result)
