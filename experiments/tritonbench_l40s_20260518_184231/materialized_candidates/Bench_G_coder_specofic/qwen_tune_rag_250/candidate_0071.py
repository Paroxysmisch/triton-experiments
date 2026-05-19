import triton
import triton.language as tl
import torch
from torch import Tensor

@triton.jit
def square_kernel(
    in_ptr: tl.tensor, out_ptr: tl.tensor,
    stride_in_row: int, stride_out_row: int,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
):
    # Square kernel implementation here

def square(x: Tensor) -> Tensor:
    # Square function implementation here
