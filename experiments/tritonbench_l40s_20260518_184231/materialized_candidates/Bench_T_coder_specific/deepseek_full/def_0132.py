import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def mul_sub_kernel(input, other_mul, other_sub, alpha, out):
    # Triton kernel to perform element-wise multiplication and subtraction
    idx = tl.arange(0, 1)
    input_val = tl.load(input + idx)
    other_mul_val = tl.load(other_mul + idx)
    other_sub_val = tl.load(other_sub + idx)
    result = (input_val * other_mul_val) - (alpha * other_sub_val)
    tl.store(out + idx, result)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None) -> Tensor:
    # Wrapper function for mul_sub_kernel
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape

    assert input.is_contiguous()
    assert other_mul.is_contiguous()
    assert other_sub.is_contiguous()
    assert out.is_contiguous()

    assert input.shape == other_mul.shape
    assert input.shape == other_sub.shape
    assert input.shape == out.shape

    grid = (1,)
    mul_sub_kernel[grid](input, other_mul, other_sub, alpha, out)
    return out
