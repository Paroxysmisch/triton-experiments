import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,   # Input vector 1 pointer
    y_ptr,   # Input vector 2 pointer
    output_ptr,  # Output vector pointer
    n_elements,  # Total elements in vectors
    BLOCK_SIZE: tl.constexpr,  # Elements processed per block
):
    # Get 1D block index
    pid = tl.program_id(axis=0)
    # Calculate block start index
    block_start = pid * BLOCK_SIZE
    # Generate element offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to avoid out-of-bounds accesses
    mask = offsets < n_elements
    
    # Load masked blocks from both inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Compute element-wise sum
    output = x + y
    # Store masked results to output
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # Create output tensor with same shape as inputs
    output = torch.empty_like(x)
    # Verify all tensors are CUDA tensors
    assert x.is_cuda and y.is_cuda and output.is_cuda
    
    # Calculate total elements and grid size
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel with dynamic grid and block size
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output
