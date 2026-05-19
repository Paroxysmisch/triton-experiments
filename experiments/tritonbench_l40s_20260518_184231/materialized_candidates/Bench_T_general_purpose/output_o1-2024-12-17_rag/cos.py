import torch
import triton
import triton.language as tl

@triton.jit
def cos_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.cos(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def cos(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    assert input.is_cuda, "Input tensor must be on CUDA device."
    N = input.numel()
    if out is None:
        out = torch.empty_like(input)

    BLOCK_SIZE = 1024
    grid = ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    cos_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
