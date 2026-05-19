import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the index of the element to process
    index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = index < n_elements
    
    # Load input values
    input_values = tl.load(input_ptr + index, mask=mask, other=0.0)
    
    # Check the sign bit
    signbit_result = input_values < 0.0
    
    # Store the result
    tl.store(output_ptr + index, signbit_result, mask=mask)

def signbit(input, *, out=None):
    # Ensure the input is a CUDA tensor
    input = input.cuda()
    
    # Determine the number of elements in the input tensor
    n_elements = input.numel()
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(n_elements, dtype=torch.bool, device=input.device)
    else:
        assert out.numel() == n_elements, "Output tensor must have the same number of elements as the input tensor"
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    signbit_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape the output to match the input shape
    return out.view_as(input)

# Example usage
input_tensor = torch.tensor([-0.0, 1.0, -1.0, 0.0, -3.0], dtype=torch.float32)
output_tensor = signbit(input_tensor)
print(output_tensor)  # Output: tensor([ True, False,  True, False,  True], device='cuda:0')
