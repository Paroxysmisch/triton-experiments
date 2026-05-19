import triton
import triton.language as tl
import torch

@triton.jit
def _leaky_relu_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    negative_slope, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.where(x >= 0, x, negative_slope * x)
    tl.store(out_ptr + offsets, out, mask=mask)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if not torch.is_tensor(input):
        raise TypeError("Input must be a torch.Tensor.")
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)

    n_elements = input.numel()
    # Launch configuration
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    _leaky_relu_kernel[grid](
        input, 
        out, 
        n_elements, 
        negative_slope,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
