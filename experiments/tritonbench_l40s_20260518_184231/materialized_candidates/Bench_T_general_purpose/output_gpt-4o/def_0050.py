import triton
import triton.language as tl
import torch

@triton.jit
def sqrt_exp_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program ID and range
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input elements
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)

    # Compute the square root and exponential
    sqrt_vals = tl.sqrt(input_vals)
    exp_vals = tl.exp(sqrt_vals)

    # Store the results
    tl.store(output_ptr + offsets, exp_vals, mask=mask)

def sqrt_exp(input, out=None):
    # Ensure input is a torch tensor
    assert isinstance(input, torch.Tensor), "Input must be a torch tensor"

    # If out is not provided, create a new tensor
    if out is None:
        out = torch.empty_like(input)

    # Get the number of elements in the input
    n_elements = input.numel()

    # Define the block size
    BLOCK_SIZE = 1024

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sqrt_exp_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out

# Example usage:
input_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0], device='cuda')
output_tensor = sqrt_exp(input_tensor)
print(output_tensor)  # Expected to print tensor([e^1, e^2, e^3, e^4])
