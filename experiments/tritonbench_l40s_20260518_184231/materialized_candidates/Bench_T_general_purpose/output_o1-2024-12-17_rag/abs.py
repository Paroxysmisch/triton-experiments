import torch
import triton
import triton.language as tl

@triton.jit
def abs_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    out_vals = tl.abs(vals)
    tl.store(out_ptr + offsets, out_vals, mask=mask)

def abs(input: torch.Tensor, *, out: torch.Tensor=None) -> torch.Tensor:
    device = input.device
    assert input.is_cuda, "Input tensor must be on CUDA device."

    if out is None:
        out = torch.empty_like(input, device=device)
    else:
        assert out.is_cuda, "Output tensor must be on CUDA device."
        assert out.shape == input.shape, "Output tensor must have the same shape as input."

    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    abs_kernel[grid](
        in_ptr=input,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
