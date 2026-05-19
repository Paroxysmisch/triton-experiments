import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def sin_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program's unique index
    pid = tl.program_id(0)
    
    # Compute the start and end index for this program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds memory accesses
    mask = offsets < n_elements
    
    # Load input elements from memory
    x = tl.load(in_ptr0 + offsets, mask=mask, other=0.0)
    
    # Compute sine
    y = tl.sin(x)
    
    # Store the result
    tl.store(out_ptr + offsets, y, mask=mask)

# Define the Python wrapper function
def sin_triton(input_tensor, block_size=1024):
    # Ensure the input is a torch tensor and is on the GPU
    assert input_tensor.is_cuda, "Input tensor must be on the GPU"
    
    # Create an output tensor of the same shape and type
    output_tensor = torch.empty_like(input_tensor)
    
    # Get the number of elements
    n_elements = input_tensor.numel()
    
    # Calculate the number of blocks needed
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    sin_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=block_size)
    
    return output_tensor

# Example usage:
# input_tensor = torch.randn(1024, device='cuda')
# output_tensor = sin_triton(input_tensor)
# print(output_tensor)
