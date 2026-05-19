import torch
import triton
import triton.language as tl

@triton.jit
def log1p_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask)
    output_x = tl.math.log(x + 1)
    tl.store(output + offsets, output_x, mask=mask)

def log1p(input, *, out=None):
    assert (
        input.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    ), "Only float types are supported"
    assert input.is_contiguous(), "Input must be contiguous"
    if out == None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()
    assert out.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ], "Only float types are supported"
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    log1p_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
