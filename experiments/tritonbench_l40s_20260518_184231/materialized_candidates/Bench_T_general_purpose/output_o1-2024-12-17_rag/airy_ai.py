import torch
import triton
import triton.language as tl

@triton.jit
def _airy_ai_kernel(
    inp_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)

    # Example polynomial approximation for Ai(x) around x=0, for demonstration only
    # Ai(x) ≈ c0 + c1*x + c2*x^2 + c3*x^3
    c0 = 0.35502805388
    c1 = -0.25881940379
    c2 = 0.020619652471
    c3 = -0.00133390876
    y = c0 + x*(c1 + x*(c2 + x*c3))

    tl.store(out_ptr + offsets, y, mask=mask)

def airy_ai(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    assert input.is_cuda and out.is_cuda, "Tensors must be on CUDA device."

    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    _airy_ai_kernel[grid](
        inp_ptr=input,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
