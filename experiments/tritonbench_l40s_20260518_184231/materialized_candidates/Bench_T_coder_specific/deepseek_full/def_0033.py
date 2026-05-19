import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def logsumexp(input, dim, keepdim, out):
    # Triton kernel to compute the logsumexp
    input_arg = input.to(tl.float32)
    output_dtype = out.dtype
    if output_dtype == tl.float16:
        input_arg = input.to(tl.float16)
    input_arg = tl.where(tl.isfinite(input_arg), input_arg, tl.full_like(input_arg, 0.0))
    input_arg = input_arg - tl.max(input_arg, 0)
    numerator = tl.exp(input_arg)
    denominator = tl.sum(numerator, 0)
    result = tl.log(denominator)
    if not keepdim:
        result = tl.squeeze(result, dim)
    return result

def logsumexp(input: Tensor, dim: int = None, keepdim: bool = False, *, out: Tensor = None) -> Tensor:
    # Wrapper function for torch.logsumexp
    if dim is None:
        input = input.contiguous()
        dim = list(range(input.ndim))
        keepdim = True
    if out is None:
        out = torch.empty_like(input)
    input_arg = input.contiguous()
    out_arg = out.contiguous()
    out_arg.copy_(input_arg)
    out_arg = logsumexp(out_arg, dim, keepdim)
    out_arg.copy_(input_arg)
    return out_arg
