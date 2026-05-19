import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def sin_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the block index for this program instance
    pid = tl.program_id(axis=0)
    
    # Calculate the starting index for this block
    start_idx = pid * BLOCK_SIZE
    
    # Create a range of offsets for the elements in this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input data, applying the mask
    x = tl.load(in_ptr0 + offsets, mask=mask)
    
    # Compute the sine of the input elements
    y = tl.sin(x)
    
    # Store the result back to the output array, using the mask
    tl.store(out_ptr + offsets, y, mask=mask)

# Define the wrapper function
def sin_triton(x):
    # Convert input to a torch tensor if it's not already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Ensure the input tensor is on the GPU
    x = x.cuda()
    
    # Create an output tensor of the same shape
    y = torch.empty_like(x)
    
    # Get the number of elements in the input tensor
    n_elements = x.numel()
    
    # Define the block size
    BLOCK_SIZE = 4
    
    # Calculate the number of blocks needed
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    sin_kernel[grid_size](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return y

# Example usage
x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).cuda()
y = sin_triton(x)
print(y.cpu())  # Move the result back to CPU for printing
