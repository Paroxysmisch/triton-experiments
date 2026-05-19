import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    input_ptr, 
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x_tanh = tl.libdevice.tanh(x)
    tl.store(output_ptr + offsets, x_tanh, mask=mask)

def tanh(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    assert input.is_cuda and out.is_cuda, "Input and output must be on GPU."

    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    tanh_kernel[grid](
        input, 
        out, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
