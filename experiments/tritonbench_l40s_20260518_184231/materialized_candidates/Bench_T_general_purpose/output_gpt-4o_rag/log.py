import torch
import triton
import triton.language as tl

@triton.jit
def log_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Obtain the index of the first element this program instance should process
    index = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = index < n_elements
    # Load the input data
    x = tl.load(input_ptr + index, mask=mask)
    # Compute the natural logarithm
    y = tl.log(x)
    # Store the result
    tl.store(output_ptr + index, y, mask=mask)

def log(input, *, out=None):
    # Ensure input is a contiguous tensor
    input = input.contiguous()
    # Determine the number of elements
    n_elements = input.numel()
    # Set block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    # If out is not provided, create a new tensor
    if out is None:
        out = torch.empty_like(input)
    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    log_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
