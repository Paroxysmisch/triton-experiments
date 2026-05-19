import torch
import triton
import triton.language as tl

@triton.jit
def log1p_kernel(output_ptr, input_ptr, n_elements):
    # Compute the index of the current element
    idx = tl.program_id(0)
    # Load the input value
    x = tl.load(input_ptr + idx)
    # Compute log1p: log(1 + x)
    y = tl.log1p(x)
    # Store the result
    tl.store(output_ptr + idx, y)

def log1p(input, *, out=None):
    # Determine the number of elements in the input tensor
    n_elements = input.numel()
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    # Launch the kernel
    log1p_kernel[(n_elements,)](
        out,
        input,
        n_elements
    )
    return out
