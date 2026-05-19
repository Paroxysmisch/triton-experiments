import triton
import torch

@torch.jit.script
def reciprocal(input, out=None):
    if out is None:
        out = torch.empty_like(input)

    # Ensure input is in the correct dtype (default scalar type)
    if input.dtype.is_integer:
        input = input.to(torch.float32)

    # Get the number of elements
    n_elements = input.numel()

    # Allocate memory for the output tensor
    if out.device != input.device or out.shape != input.shape:
        out = torch.empty_like(input, device=input.device, requires_grad=False)

    # Launch the kernel
    threads_per_block = 256
    blocks_per_grid = (n_elements + threads_per_block - 1) // threads_per_block
    reciprocal_kernel[blocks_per_grid, threads_per_block](input.data_ptr(), out.data_ptr(), n_elements)

    return out
