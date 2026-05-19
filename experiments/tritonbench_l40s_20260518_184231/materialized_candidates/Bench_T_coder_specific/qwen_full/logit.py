import torch
import triton
import triton.language as tl

@triton.jit
def logit(input, eps=None):
    if eps is None:
        result = tl.where(input < 0, float("nan"), input)
        result = tl.where(input > 1, float("nan"), result)
    else:
        input = tl.where(input < eps, eps, input)
        input = tl.where(input > 1 - eps, 1 - eps, input)
    odd = input * 2 - 1
    result = tl.math.log(odd / (1 - odd))
    return result

def torch_logit(input, eps=None, *, out=None):
    return logit(input, eps)
