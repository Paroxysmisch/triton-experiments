import triton
import triton.language as tl
import torch

@triton.jit
def kldivergence_kernel(
    x_ptr, 
    y_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=1)

    # KL divergence for each element: x * log(x / y)
    output = x * tl.log(x / y)

    tl.store(output_ptr + offsets, output, mask=mask)

def kldivergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("Both x and y must be on CUDA.")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.dtype != y.dtype:
        raise ValueError("x and y must have the same dtype.")

    n_elements = x.numel()
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = 1024

    kldivergence_kernel[grid](
        x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    return output
