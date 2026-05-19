import triton
import triton.language as tl
import torch

# Triton kernel
@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program instance determines its offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements
    # Load data from x_ptr, perform computation, and store result in output_ptr
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.math.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function to call the Triton kernel
def call_kernel(x):
    n_elements = x.numel()
    output = torch.empty_like(x)
    # Define a grid lambda to encapsulate the block calculations
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # Launch the Triton kernel with the grid, input, output, and n_elements as arguments
    kernel_function[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

# Example usage
x = torch.randn(1024, device='cuda')
output = call_kernel(x)
