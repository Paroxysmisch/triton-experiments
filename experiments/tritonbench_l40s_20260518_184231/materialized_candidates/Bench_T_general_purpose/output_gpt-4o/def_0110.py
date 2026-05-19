import triton
import triton.language as tl

@triton.jit
def exp_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Obtain the program id and calculate the offset for this block
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask to handle out-of-bounds accesses
    mask = offsets < n_elements

    # Load input elements, apply exponential, and store the result
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    exp_vals = tl.exp(input_vals)
    tl.store(output_ptr + offsets, exp_vals, mask=mask)

import torch

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Allocate memory for the output of the exponential operation
    exp_output = torch.empty_like(input, dtype=dtype if dtype else input.dtype)

    # Calculate the number of elements in the input tensor
    n_elements = input.numel()

    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # Define a block size for the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    exp_kernel[grid](input, exp_output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Compute the mean of the exponential values
    if dim is None:
        result = exp_output.mean(dtype=dtype)
    else:
        result = exp_output.mean(dim=dim, keepdim=keepdim, dtype=dtype)

    # If an output tensor is provided, copy the result into it
    if out is not None:
        out.copy_(result)
        return out

    return result
