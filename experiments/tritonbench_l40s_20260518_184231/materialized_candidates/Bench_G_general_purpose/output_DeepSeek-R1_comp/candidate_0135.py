import triton
import triton.language as tl
import torch

@triton.jit
def kernel_function(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant for optimization
):
    # Determine which block this program instance represents
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create range of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds memory accesses
    mask = offsets < n_elements
    
    # Load data from input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute element-wise sine
    output = tl.math.sin(x)
    
    # Store results to output tensor
    tl.store(output_ptr + offsets, output, mask=mask)

def call_kernel(x: torch.Tensor):
    # Get total number of elements in input tensor
    n_elements = x.numel()
    
    # Initialize output tensor with same shape/dtype
    output = torch.empty_like(x)
    
    # Define grid configuration strategy
    grid = lambda meta: (
        triton.cdiv(n_elements, meta['BLOCK_SIZE']),  # Calculate needed blocks
    )
    
    # Launch kernel with optimal block size
    block_size = triton.next_power_of_2(n_elements)
    kernel_function[grid](
        x, output, n_elements,
        BLOCK_SIZE=block_size
    )
    
    return output
