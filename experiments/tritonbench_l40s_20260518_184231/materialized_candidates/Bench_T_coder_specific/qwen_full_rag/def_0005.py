import triton
import triton.language as tl
import torch
from torch import Tensor

@triton.jit
def _relu_sqrt(x, x_original):
    x = tl.where(x >= 0, x, 0)
    x = tl.math.sqrt(x)
    return x, x_original

def relu_sqrt(input: Tensor, inplace: bool = False, out: Tensor = None) -> Tensor:
    if not inplace:
        if out is None:
            out = torch.empty_like(input, dtype=input.dtype)
        else:
            assert out.shape == input.shape and out.dtype == input.dtype
        out.copy_(input)
        N = out.numel()
        BLOCK_SIZE = 128
        grid_size = triton.cdiv(N, BLOCK_SIZE)
        triton_kernel = lambda meta: _relu_sqrt[grid_size, 1, 1](out, out)
        triton_compile_and_cache(triton_kernel, key="relu_sqrt", warm_cache_only=True)
    else:
        assert out is None
        N = input.numel()
        BLOCK_SIZE = 128
        grid_size = triton.cdiv(N, BLOCK_SIZE)
        triton_kernel = lambda meta: _relu_sqrt[grid_size, 1, 1](input, input)
        triton_compile_and_cache(
            triton_kernel, key="relu_sqrt_inplace", warm_cache_only=True
        )
