import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A_ptr, B_ptr, C_ptr, size, BLOCK: tl.constexpr):
    # Get the program index for the current block
    prog_id = tl.program_id(0)
    
    # Calculate the offsets for the current block
    offsets = prog_id * BLOCK + tl.arange(0, BLOCK)
    
    # Create a mask to handle boundary conditions
    mask = offsets < size
    
    # Load values from input tensors A and B
    A_values = tl.load(A_ptr + offsets, mask=mask)
    B_values = tl.load(B_ptr + offsets, mask=mask)
    
    # Compute the element-wise addition
    C_values = A_values + B_values
    
    # Store the result in the output tensor C
    tl.store(C_ptr + offsets, C_values, mask=mask)

def custom_add(a: torch.Tensor, b: torch.Tensor, BLOCK=16):
    # Ensure the input tensors have the same shape
    assert a.shape == b.shape, "Input tensors must have the same shape."
    
    # Create an empty tensor for the result with the same shape as the input tensors
    c = torch.empty_like(a)
    
    # Get the total number of elements in the input tensors
    size = a.numel()
    
    # Ensure the input tensors are on the GPU
    assert a.is_cuda and b.is_cuda and c.is_cuda, "Tensors must be on GPU."
    
    # Define the grid size for launching the kernel
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK']),)
    
    # Launch the Triton kernel
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    
    # Return the result tensor
    return c

# Example usage
BLOCK_SIZE = 16
size = int(1e5)
a = torch.rand(size, device='cuda')
b = torch.rand(size, device='cuda')
torch.cuda.synchronize()
out = custom_add(a, b, BLOCK_SIZE)
