import triton
import triton.language as tl

@triton.jit
def exp_mean_kernel(input_ptr, output_ptr, stride, numel, BLOCK_SIZE: tl.constexpr):
    # Compute the block and thread indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input data
    mask = offsets < numel
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Apply the exponential function
    exp_data = tl.exp(input_data)

    # Compute the mean
    mean_value = tl.sum(exp_data, axis=0) / numel

    # Store the result
    tl.store(output_ptr + pid, mean_value, mask=mask)

import torch

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
    # Convert input to the desired dtype if specified
    if dtype is not None:
        input = input.to(dtype)

    # If no dimension is specified, flatten the input tensor
    if dim is None:
        input = input.view(-1)
        dim = 0

    # Calculate the number of elements along the specified dimension
    numel = input.size(dim)

    # Prepare output tensor
    if out is None:
        out_shape = list(input.shape)
        if not keepdim:
            out_shape.pop(dim)
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    # Launch Triton kernel
    grid = (triton.cdiv(input.numel(), 1024),)
    exp_mean_kernel[grid](input, out, input.stride(dim), numel, BLOCK_SIZE=1024)

    return out
