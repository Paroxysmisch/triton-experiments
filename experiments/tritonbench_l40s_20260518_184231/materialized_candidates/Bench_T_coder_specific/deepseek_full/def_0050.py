import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def sqrt_exp(input, out=None) -> Tensor:
    # The function computes the square root of each element in input, and then applies the exponential function to the square-rooted values.
    input_sqrt = tl.sqrt(input)
    out = tl.exp(input_sqrt)
    return out

def sqrt_exp(input: Tensor, out: Tensor = None) -> Tensor:
    # The function computes the square root of each element in input, and then applies the exponential function to the square-rooted values.
    if out is None:
        out = torch.empty_like(input)

    input_reshaped = input.reshape(-1)
    out_reshaped = sqrt_exp(input_reshaped)
    out = out_reshaped.reshape(input.shape)

    return out
