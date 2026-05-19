import torch
import triton
import triton.language as tl

@triton.jit
def mul_relu_kernel(input, other, out):
    # Triton kernel for element-wise multiplication and ReLU
    idx = tl.arange(0, 1)[:, None] * tl.arange(0, 1)[None, :]
    inp = tl.load(input + idx)
    ot = tl.load(other + idx)
    result = tl.relu(inp * ot)
    tl.store(out + idx, result)

def mul_relu(input, other, inplace=False, out=None) -> torch.Tensor:
    # Wrapper function for element-wise multiplication and ReLU
    if out is None:
        if inplace:
            out = input
        else:
            out = torch.empty_like(input)

    input = input.contiguous()
    other = other.contiguous()
    out = out.contiguous()

    assert input.is_contiguous()
    assert other.is_contiguous()
    assert out.is_contiguous()

    grid = (1,)
    mul_relu_kernel[grid](input, other, out)
    return out
