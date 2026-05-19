import torch
import triton
import triton.language as tl
from .utils import tile
from .exp import fused_exp

@triton.jit
def fused_tile_exp(input, dims, **META):
    # Tiling operation
    tiled = tile(input, dims)
    META['TILE_INPUT_DIMS'] = dims
    # Exponential function
    return fused_exp(tiled)

def tiled_exp(input: torch.Tensor, dims: tuple, *, out=None) -> torch.Tensor:
    # Ensure that the input tensor is on the CUDA device
    if not input.is_cuda:
        input = input.cuda()
    # Call the Triton wrapper function
    return fused_tile_exp(input, dims, out=out)
