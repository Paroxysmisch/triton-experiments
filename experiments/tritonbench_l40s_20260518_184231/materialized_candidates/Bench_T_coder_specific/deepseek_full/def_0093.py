import torch
import triton
import triton.language as tl

@triton.jit
def softmax_log(input, dim=-1, dtype=None):
    # Apply natural logarithm
    input = tl.log(input)
    # Apply softmax
    input -= tl.max(input, axis=dim, keepdims=True)
    input = tl.exp(input)
    input /= tl.sum(input, axis=dim, keepdims=True)
    return input

def softmax_log(input, dim=-1, dtype=None):
    # Convert input to Tensor if it is not already
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    # Cast input to desired dtype if provided
    if dtype is not None:
        input = input.to(dtype)
    # Define function signature for triton
    triton_softmax_log = softmax_log.warp(input, dim=dim)
    # Call triton function
    return triton_softmax_log(input, dim=dim)
