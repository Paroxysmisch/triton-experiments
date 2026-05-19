import torch
import triton
from triton.util import to_cuda

@torch.jit.script
def div(input, other, *, rounding_mode=None, out=None):
    if out is None:
        out = torch.empty_like(input)

    assert input.device.type == 'cuda', "Input must be on CUDA"
    assert other.device.type == 'cuda', "Other must be on CUDA"

    n_elements = input.numel()
    block_size_x = 256
    grid_size_x = (n_elements + block_size_x - 1) // block_size_x

    # Allocate memory for the output tensor
    output = torch.zeros_like(input)

    # Launch the Triton kernel
    div_kernel[grid_size_x, block_size_x](to_cuda(input), to_cuda(other), to_cuda(output), n_elements)

    return output
