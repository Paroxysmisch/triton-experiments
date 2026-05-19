import torch
import F

@triton.jit
def mul_relu_kernel(input, other, out, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask)
    y = tl.load(other + offsets, mask=mask)
    mul = x * y
    out = F.relu(mul)
    tl.store(out + offsets, out, mask=mask)

def mul_relu(input, other, inplace=False, out=None) -> torch.Tensor:
    assert input.is_contiguous()
    assert other.is_contiguous()
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    mul_relu_kernel[grid](input, other, out, n_elements, BLOCK_SIZE=1024)
    return out
