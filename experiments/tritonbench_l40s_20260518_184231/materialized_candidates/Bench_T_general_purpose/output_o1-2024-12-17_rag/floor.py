import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def floor_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    floored = tl.floor(x)
    tl.store(output_ptr + offsets, floored, mask=mask)


def floor(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # If input is not floating-point, return a copy (or copy into out if provided)
    if not input.is_floating_point():
        if out is None:
            return input.clone()
        else:
            return out.copy_(input)

    # Prepare output
    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        assert out.is_cuda, "Out tensor must be on CUDA device."
        assert out.shape == input.shape, "Out tensor must have the same shape as input."
        assert out.dtype == input.dtype, "Out tensor must have the same dtype as input."

    # Ensure input is on CUDA
    assert input.is_cuda, "Input tensor must be on CUDA device."

    # Flatten input and output for 1D kernel
    flat_input = input.view(-1)
    flat_out = out.view(-1)
    n_elements = flat_input.numel()

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    floor_kernel[grid](
        input_ptr = flat_input,
        output_ptr = flat_out,
        n_elements = n_elements,
        BLOCK_SIZE = BLOCK_SIZE
    )

    return out
