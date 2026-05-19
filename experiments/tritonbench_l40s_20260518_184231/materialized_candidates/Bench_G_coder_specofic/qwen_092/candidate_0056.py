import triton
import triton.language as tl
import torch

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    # Get the program ID and the offset within the block
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Ensure offsets are within bounds
    offsets = offsets[offs < size]
    
    # Load data from A and B
    a = tl.load(A + offsets)
    b = tl.load(B + offsets)
    
    # Perform addition
    c = a + b
    
    # Store the result in C
    tl.store(C + offsets, c)

def custom_add(a, b):
    # Ensure inputs are PyTorch tensors
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    
    # Create an empty tensor for the result
    c = torch.empty_like(a)
    
    # Get the size of the tensors
    size = a.numel()
    
    # Set the block size
    BLOCK = 16
    
    # Calculate the grid size
    grid = triton.cdiv(size, BLOCK)
    
    # Launch the Triton kernel
    _add_kernel[grid, BLOCK](a, b, c, size, BLOCK)
    
    return c

# Example usage
a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float32)

result = custom_add(a, b)
print(result)
