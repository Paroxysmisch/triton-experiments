import triton
import triton.language as tl

@triton.jit
def sin_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the PID (program ID) and the range of elements this program instance will process
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to guard memory operations
    mask = offsets < n_elements
    
    # Load input elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    
    # Compute the sine of each element
    y = tl.sin(x)
    
    # Store the results
    tl.store(out_ptr + offsets, y, mask=mask)

import triton
import triton.runtime
import torch

def sin_triton(input_tensor, output_tensor, block_size=1024):
    # Ensure the input and output tensors are on the same device
    assert input_tensor.device == output_tensor.device
    device = input_tensor.device
    
    # Get the number of elements
    n_elements = input_tensor.numel()
    
    # Launch the kernel
    grid = (triton.cdiv(n_elements, block_size),)
    sin_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE=block_size)
    
    # Synchronize to ensure the kernel has finished
    torch.cuda.synchronize(device)

import torch

# Create input and output tensors
input_tensor = torch.randn(1024, device='cuda')
output_tensor = torch.empty_like(input_tensor)

# Call the wrapper function
sin_triton(input_tensor, output_tensor)

# Print the results
print(output_tensor)
