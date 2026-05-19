import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    in_ptr0, in_ptr1, out_ptr,  # Pointers to tensors
    n_elements,                  # Total elements to process
    BLOCK_SIZE: tl.constexpr,     # Elements per block
):
    # Determine block index
    pid = tl.program_id(axis=0)
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    # Generate range of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to prevent out-of-bounds access
    mask = offsets < n_elements
    
    # Load input values with mask
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    # Compute element-wise sum
    output = x + y
    # Store results with mask
    tl.store(out_ptr + offsets, output, mask=mask)

def add_wrapper(x: torch.Tensor, y: torch.Tensor):
    # Create output tensor
    out = torch.zeros_like(x)
    # Get total number of elements
    n_elements = x.numel()
    
    # Configure block size (can be tuned)
    BLOCK_SIZE = 1024
    # Calculate number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with appropriate grid
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)
    
    return out

# Example usage:
# x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
# y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
# result = add_wrapper(x, y)
