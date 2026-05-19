import torch
import triton
import triton.language as tl

@triton.jit
def _rsqrt_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = 1.0 / tl.sqrt(x)  # elementwise reciprocal square root
    tl.store(output_ptr + offsets, y, mask=mask)

def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    contracted = torch.tensordot(a, b, dims=dims)
    out = torch.empty_like(contracted)
    n_elements = contracted.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _rsqrt_kernel[grid](
        contracted, 
        out, 
        n_elements, 
        BLOCK_SIZE
    )
    return out
