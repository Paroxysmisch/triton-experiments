import torch
import triton
import triton.language as tl

@triton.jit
def i0_kernel(
    input_ptr, 
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr, 
    MAX_ITER: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x

    # Initialize the series
    term = 1.0
    res = 1.0

    # Compute the I0 series expansion up to MAX_ITER
    for k in range(1, MAX_ITER + 1):
        term = term * (x_sq / 4.0) / (k * k)
        res = res + term

    tl.store(output_ptr + offsets, res, mask=mask)


def i0(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    device = input.device
    if out is None:
        out = torch.empty_like(input, device=device)

    n = input.numel()
    assert input.is_cuda and out.is_cuda, 'Tensors must be on CUDA device'

    BLOCK_SIZE = 1024
    grid = ( (n + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    MAX_ITER = 25

    i0_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_ITER=MAX_ITER
    )

    return out
