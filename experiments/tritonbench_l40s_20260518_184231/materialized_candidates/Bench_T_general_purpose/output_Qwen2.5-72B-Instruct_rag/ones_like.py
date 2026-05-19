import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

# Triton kernel that sets all elements in the output tensor to 1.0
@triton.jit
def ones_kernel(
    output_ptr,  # Pointer to the output tensor in GPU memory
    n_elements,  # Total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Size of each block of threads
):
    pid = tl.program_id(axis=0)  # Get the block index
    block_start = pid * BLOCK_SIZE  # Calculate the start index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Calculate offsets for each thread
    mask = offsets < n_elements  # Mask to ensure we don't write out of bounds
    tl.store(output_ptr + offsets, 1.0, mask=mask)  # Store 1.0 in all valid positions

# Function to initialize a tensor of the same size as the input tensor with ones using the Triton kernel
def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    # Determine the properties of the output tensor based on the input tensor and provided arguments
    if dtype is None:
        dtype = input.dtype
    if layout is None:
        layout = input.layout
    if device is None:
        device = input.device
    if memory_format is None:
        memory_format = input.memory_format

    # Create an empty tensor with the same size as the input tensor
    out = torch.empty_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)
    
    # Calculate the total number of elements
    N = volume(input.shape)
    
    # Determine the grid size
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    
    # Launch the Triton kernel
    with torch.cuda.device(device):
        ones_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    
    return out  # Return the initialized tensor

# Example usage
input_tensor = torch.tensor([1, 2, 3], device='cuda')
output_tensor = ones_like(input_tensor)
print(output_tensor)
