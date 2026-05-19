import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def rsqrt_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute reciprocal sqrt, return nan for negative values
    is_neg = x < 0.0
    x_sqrt = tl.sqrt(x)
    rsqrt_val = 1.0 / x_sqrt
    out_val = tl.where(is_neg, float('nan'), rsqrt_val)
    tl.store(output_ptr + offsets, out_val, mask=mask)

def rsqrt(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)

    assert input.is_cuda and out.is_cuda, 'Tensors must be on GPU'
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    rsqrt_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
