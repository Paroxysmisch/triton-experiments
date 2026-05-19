# triton_kernel.py

import triton
import triton.language as tl

@triton.jit
def exp_kernel(input_ptr, output_ptr, n_elements):
    # Define the grid size
    pid = tl.program_id(0)
    # Calculate the index for the current thread
    index = pid * tl.num_warps() + tl.arange(0, tl.num_warps())
    # Ensure we do not exceed the number of elements
    mask = index < n_elements
    # Load input tensor elements
    x = tl.load(input_ptr + index, mask=mask)
    # Compute the exponential
    y = tl.exp(x)
    # Store the result in the output tensor
    tl.store(output_ptr + index, y, mask=mask)

def exp(input, *, out=None):
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    # Allocate output tensor if not provided
    if out is None:
        out = input.new_empty(input.shape)
    # Launch the kernel
    grid = (triton.cdiv(n_elements, triton.num_warps()),)
    exp_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements)
    return out
