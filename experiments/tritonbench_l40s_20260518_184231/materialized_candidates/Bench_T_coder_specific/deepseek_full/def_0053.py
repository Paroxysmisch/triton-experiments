import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def mul_relu(input, other, inplace, out):
    # Element-wise multiplication
    input_mul_other = torch.mul(input, other)
    # Apply the ReLU function
    out = tl.where(input_mul_other > 0, input_mul_other, 0)
    return out

def mul_relu(input, other, inplace=False, out=None) -> Tensor:
    if inplace:
        raise NotImplementedError("In-place operation is not supported.")
    else:
        return mul_relu(input, other, inplace, out)
