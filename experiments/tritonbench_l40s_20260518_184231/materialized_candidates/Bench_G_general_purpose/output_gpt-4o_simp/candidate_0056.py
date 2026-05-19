import torch
import triton
import triton.language as tl

# Define the Triton kernel for element-wise addition
@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    # Get the program ID for this instance
    prog_id = tl.program_id(0)
    
    # Calculate the start offset for this block
    offsets = prog_id * BLOCK + tl.arange(0, BLOCK)
    
    # Create a mask to handle boundary conditions
    mask = offsets < size
    
    # Load data from input tensors A and B with boundary check
    a_vals = tl.load(A + offsets, mask=mask, other=0.0)
    b_vals = tl.load(B + offsets, mask=mask, other=0.0)
    
    # Perform element-wise addition
    c_vals = a_vals + b_vals
    
    # Store the result in output tensor C
    tl.store(C + offsets, c_vals, mask=mask)

# Define the wrapper function for the Triton kernel
def custom_add(a, b):
    # Ensure input tensors are of the same size
    assert a.size() == b.size(), "Input tensors must have the same size"
    
    # Determine the size of the tensors
    size = a.numel()
    
    # Create an output tensor of the same size
    c = torch.empty_like(a)
    
    # Define the block size for the kernel
    BLOCK = 16
    
    # Calculate the grid size needed to cover the entire tensor
    grid = (size + BLOCK - 1) // BLOCK  # This ensures we cover the entire tensor
    
    # Launch the Triton kernel
    _add_kernel[grid](a, b, c, size, BLOCK)
    
    # Return the result tensor
    return c

# Example usage
if __name__ == "__main__":
    # Create example tensors
    a = torch.randn(128, device='cuda')
    b = torch.randn(128, device='cuda')
    
    # Perform custom addition
    c = custom_add(a, b)
    
    # Print the result
    print(c)
