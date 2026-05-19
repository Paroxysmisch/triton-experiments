import torch
import triton
import triton.language as tl

# Triton kernel for element-wise addition
@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    # Get the program index
    pid = tl.program_id(0)
    
    # Calculate the offsets for block-wise data processing
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Create a mask to handle out-of-bound accesses
    mask = offs < size
    
    # Load data from A and B with the mask
    a_vals = tl.load(A + offs, mask=mask)
    b_vals = tl.load(B + offs, mask=mask)
    
    # Compute the sum
    c_vals = a_vals + b_vals
    
    # Store the result in C with the same mask
    tl.store(C + offs, c_vals, mask=mask)

# Wrapper function to invoke the Triton kernel
def custom_add(a: torch.Tensor, b: torch.Tensor):
    # Ensure the input tensors have the same shape
    assert a.shape == b.shape, "Input tensors must have the same shape"
    
    # Initialize the output tensor with the same shape as a
    c = torch.empty_like(a)
    
    # Compute the size of the input tensors
    size = a.numel()
    
    # Set the block size
    BLOCK = 16
    
    # Calculate the grid size
    grid = (triton.cdiv(size, BLOCK),)
    
    # Launch the kernel
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    
    return c

# Example usage
size = int(1e5)
a = torch.rand(size, device='cuda')
b = torch.rand(size, device='cuda')
torch.cuda.synchronize()

# Perform the custom addition
c = custom_add(a, b)

# Verify the result
torch.allclose(a + b, c)  # Should return True
