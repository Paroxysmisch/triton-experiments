import triton
import triton.language as tl
import torch
import math

@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define the number of elements each program should process
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data from x_ptr
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute the sine of the loaded data
    output = tl.math.sin(x)
    # Store the result in output_ptr
    tl.store(output_ptr + offsets, output, mask=mask)

def call_kernel(x, dtype):
    output = torch.empty_like(x, dtype=dtype)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    # Define the grid lambda to handle thread block calculations
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    # Launch the kernel
    kernel_function[grid](x, output, n_elements, BLOCK_SIZE)
    return output

# Example Usage
x = torch.arange(1024, device='cuda')
output = call_kernel(x, torch.float32)
print(output)
