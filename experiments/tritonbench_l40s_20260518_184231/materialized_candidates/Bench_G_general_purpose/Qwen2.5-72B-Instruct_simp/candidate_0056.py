import torch
import triton
import triton.language as tl

# Define the block size
BLOCK = 16

# Triton kernel for element-wise addition
@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    # Get the program ID
    prog_id = tl.program_id(0)
    
    # Calculate the offsets for the block
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    
    # Create a mask to handle boundary conditions
    mask = offs < size
    
    # Load the data from the input tensors with boundary checks
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    
    # Perform the addition
    c = a + b
    
    # Store the result back to the output tensor with boundary checks
    tl.store(C + offs, c, mask=mask)

# Wrapper function to add two PyTorch tensors
def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensors are on the same device
    assert a.device == b.device, "Input tensors must be on the same device"
    
    # Ensure the input tensors have the same shape
    assert a.shape == b.shape, "Input tensors must have the same shape"
    
    # Create an empty tensor to store the result
    c = torch.empty_like(a)
    
    # Determine the grid size
    size = a.numel()
    grid = (size + BLOCK - 1) // BLOCK
    
    # Launch the Triton kernel
    _add_kernel[grid](a, b, c, size, BLOCK)
    
    return c

# Example usage
if __name__ == "__main__":
    # Create two input tensors
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    
    # Perform the custom addition
    c = custom_add(a, b)
    
    # Verify the result
    expected = a + b
    assert torch.allclose(c, expected), "The result does not match the expected output"
    
    print("Custom addition is correct!")
