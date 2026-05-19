import triton
import triton.language as tl

@triton.jit
def exp_mean_kernel(input_ptr, output_ptr, n_elements, dim, keepdim, dtype, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the exponential of each element in the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_block = tl.load(input_ptr + offsets, mask=mask)
    exp_block = tl.exp(input_block)

    # Compute the mean of the exponential values
    mean_value = tl.sum(exp_block, axis=0) / n_elements

    # Store the result
    if keepdim:
        output_block = mean_value
    else:
        output_block = mean_value[None]  # Reshape to a 1D tensor

    tl.store(output_ptr + pid, output_block, mask=pid < n_elements)

import torch
import triton
import triton.language as tl

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    if dtype is not None:
        input = input.to(dtype)

    if dim is None:
        # Compute the mean over all elements
        n_elements = input.numel()
        output = torch.empty(1, device=input.device, dtype=input.dtype)
        grid = (1,)
    else:
        # Compute the mean along the specified dimension
        if dim < 0:
            dim = input.dim() + dim
        n_elements = input.size(dim)
        output_shape = list(input.shape)
        if not keepdim:
            output_shape.pop(dim)
        else:
            output_shape[dim] = 1
        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
        grid = (output.numel(),)

    # Launch the Triton kernel
    exp_mean_kernel[grid](
        input, output, n_elements, dim, keepdim, input.dtype, BLOCK_SIZE=1024
    )

    if out is not None:
        out.copy_(output)
    else:
        return output
