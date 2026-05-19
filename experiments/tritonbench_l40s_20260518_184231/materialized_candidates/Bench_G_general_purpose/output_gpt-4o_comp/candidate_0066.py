import torch
import triton
import triton.language as tl

# Define the block size for processing elements
BLOCK_SIZE = 1024

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the block index for a 1D launch grid
    block_idx = tl.program_id(axis=0)
    
    # Calculate the start index for this block
    block_start = block_idx * BLOCK_SIZE
    
    # Compute offsets within this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to ensure we don't access out-of-bounds elements
    mask = offsets < n_elements
    
    # Load elements from input vectors, apply mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute the sum
    output = x + y
    
    # Store the result in the output vector, apply mask
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Ensure input tensors are on the CUDA device
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device"
    
    # Ensure input tensors have the same shape
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    # Calculate the total number of elements
    n_elements = x.numel()
    
    # Create an output tensor on the same device and with the same dtype
    output = torch.empty_like(x)
    
    # Calculate the grid size
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the Triton kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

# Example usage
x = torch.rand(10240, device='cuda')
y = torch.rand(10240, device='cuda')
result = add(x, y)
print(result)
