import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

def sigmoid(input: Tensor, *, out=None) -> Tensor:
    if type(input) == int:
        raise TypeError("Argument `input` of `sigmoid` cannot be an int.")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("Argument `out` of `sigmoid` must be None or a torch.Tensor.")
    if out is not None and input.shape != out.shape:
        raise ValueError(f"Incompatible shapes: {input.shape} and {out.shape}")
    if out is None:
        return tanh(input * 0.5) * 0.5 + 0.5
    else:
        tanh(input * 0.5) * 0.5 + 0.5
