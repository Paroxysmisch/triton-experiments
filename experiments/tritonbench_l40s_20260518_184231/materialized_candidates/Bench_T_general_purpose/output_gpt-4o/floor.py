import triton
import triton.language as tl
import torch

@triton.jit
def floor_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the index of the current element
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the input values
    input_values = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute the floor of the input values
    floored_values = tl.floor(input_values)
    
    # Store the result
    tl.store(output_ptr + offsets, floored_values, mask=mask)

def floor(input, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    elif not isinstance(out, torch.Tensor):
        raise TypeError("Output must be a torch.Tensor or None")
    elif out.shape != input.shape:
        raise ValueError("Output tensor must have the same shape as input tensor")
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    floor_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    
    return out

# Example usage
input_tensor = torch.tensor([1.7, 2.3, -1.2, 4.0, 5.9], dtype=torch.float32)
output_tensor = floor(input_tensor)
print(output_tensor)
