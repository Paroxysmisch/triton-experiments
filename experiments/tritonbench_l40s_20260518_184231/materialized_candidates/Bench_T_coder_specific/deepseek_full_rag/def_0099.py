import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.std import _standard_deviation

@triton.jit
def gelu_none(x):
    # Convert input to float32 for better precision in operations
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the error function approximation
    return 0.5 * x_fp32 * (1 + tl.erf(x_fp32 * 0.7071067811))

@triton.jit
def gelu_tanh(x):
    # Convert input to float32 for better precision in operations
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the tanh approximation
    return (
        0.5
        * x_fp32
        * (
            1
            + tl.tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * tl.pow(x_fp32.to(tl.float32), 2)))
        )
    )

def gelu_std(
    input: Tensor,
    dim=None,
    keepdim=False,
    correction=1,
    approximate="none",
    out=None,
) -> Tensor:
    # Apply GELU activation function
    if approximate == "none":
        gelu_result = gelu_none(input)
    elif approximate == "tanh":
        gelu_result = gelu_tanh(input)
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")

    # Compute standard deviation of the result
    return _standard_deviation(
        gelu_result, dim=dim, unbiased=True, keepdim=keepdim, out=out
    ) * correction
