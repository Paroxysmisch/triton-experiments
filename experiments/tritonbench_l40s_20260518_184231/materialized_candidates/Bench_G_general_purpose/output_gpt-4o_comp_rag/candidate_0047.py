import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A_ptr, B_ptr, C_ptr, size, BLOCK: tl.constexpr):
    # Calculate the program ID for parallel execution
    prog_id = tl.program_id(axis=0)
    
    # Calculate the offsets for this block
    offsets = prog_id * BLOCK + tl.arange(0, BLOCK)
    
    # Create a mask to handle edge cases where offsets might exceed size
    mask = offsets < size
    
    # Load values from A and B using the calculated offsets and mask
    A_values = tl.load(A_ptr + offsets, mask=mask)
    B_values = tl.load(B_ptr + offsets, mask=mask)
    
    # Perform element-wise addition
    C_values = A_values + B_values
    
    # Store the result in C
    tl.store(C_ptr + offsets, C_values, mask=mask)

def custom_add(a: torch.Tensor, b: torch.Tensor):
    # Ensure that the input tensors have the same shape
    assert a.shape == b.shape, "Input tensors must have the same shape"
    
    # Initialize the output tensor
    c = torch.empty_like(a)
    
    # Calculate the total number of elements
    size = a.numel()
    
    # Define the block size
    BLOCK = 16
    
    # Calculate the grid size to ensure full coverage
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK']),)
    
    # Launch the Triton kernel
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    
    return c

# Example usage
size = int(1e5)
a = torch.rand(size, device='cuda')
b = torch.rand(size, device='cuda')
torch.cuda.synchronize()
c = custom_add(a, b)
