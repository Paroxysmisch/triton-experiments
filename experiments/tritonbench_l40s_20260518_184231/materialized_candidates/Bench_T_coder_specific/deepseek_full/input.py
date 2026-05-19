import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union

@triton.jit
def _mean(
    input,
    dim,
    keepdim,
    dtype,
    out: tl.tensor,
):
    input = input.to(dtype)
    return tl.reduce(input, dim, out, lambda x: x.mean(dim, keepdim))

def mean(
    input: Tensor,
    dim: Union[int, tuple],
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype
    if out is None:
        output_shape = list(input.shape)
        if not isinstance(dim, tuple):
            dim = (dim,)
            output_shape[dim[0]] = 1 if keepdim else dim[0]
        else:
            output_shape = [1 if d in dim else s for d, s in enumerate(input.shape)]
        out = torch.empty(output_shape, dtype=dtype, device=input.device)
    else:
        assert out.shape == input.shape, "shape of out and input must be the same"
        assert out.dtype == dtype, "dtype of out and input must be the same"
    _mean[(1,)](input, dim, keepdim, dtype, out)
    return out
